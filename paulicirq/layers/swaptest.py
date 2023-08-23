import numbers
import typing

import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow_quantum.core.ops import tfq_utility_ops

from paulicirq.gates import add_swap_test
from paulicirq.layers.addpqc import AddPQC


class SWAPTestLayer(tf.keras.layers.Layer):
    def __init__(self, state1, state2, **kwargs):
        super().__init__(**kwargs)

        qubits_in_both_states = set(state1).intersection(set(state2))
        if qubits_in_both_states:
            raise ValueError(
                "There are qubits which belong to state1 and state2 "
                f"at the same time: {qubits_in_both_states}."
            )

        self.state1 = state1
        self.state2 = state2

    def call(self, inputs):
        state1_circuit_tensor, state2_circuit_tensor = inputs

        swap_test_circuit = cirq.Circuit()
        _, _ = add_swap_test(
            state1=self.state1,
            state2=self.state2,
            circuit=swap_test_circuit,
            auxiliary_qubit=self.auxiliary_qubit,
            cswap_in_elementary_gates=True
        )
        swap_test_layer = AddPQC(model_circuit=swap_test_circuit)

        combined_input_circuit = tfq_utility_ops.append_circuit(
            state1_circuit_tensor, state2_circuit_tensor
        )
        return swap_test_layer(combined_input_circuit, append=True)

    @property
    def auxiliary_qubit(self) -> cirq.GridQubit:
        from paulicirq.utils import generate_auxiliary_qubit
        temp_circuit = cirq.Circuit([
            cirq.I.on_each(*self.state1, *self.state2)
        ])
        q_aux = generate_auxiliary_qubit(
            temp_circuit, auxiliary_qubit_type=cirq.GridQubit
        )
        return q_aux


def _inspect_state(
        state, circuit_prepend, circuit_append
) -> typing.Tuple[
    typing.Iterable[cirq.GridQubit], cirq.Circuit, cirq.Circuit
]:
    """Inspect two quantum states."""
    if state:
        if isinstance(state, cirq.GridQubit):
            state = [state]
        elif isinstance(state, typing.Iterable):
            pass
        else:
            raise TypeError(
                f"Invalid type for `state`: {type(state)}"
            )

        if isinstance(circuit_prepend, cirq.Circuit):
            if not circuit_prepend.all_qubits().issubset(state):
                raise ValueError(
                    "The circuit to be prepended contains qubits which "
                    "is not in `state`: "
                    f"qubits in `circuit_prepend` = "
                    f"{circuit_prepend.all_qubits()}; "
                    f"`state` = {state}.")
        elif circuit_prepend is None:
            circuit_prepend = cirq.Circuit()
        else:
            raise TypeError(
                f"Invalid type for `circuit_prepend`: "
                f"{type(circuit_prepend)}"
            )

        if isinstance(circuit_append, cirq.Circuit):
            if not circuit_append.all_qubits().issubset(state):
                raise ValueError(
                    "The circuit to be appended contains qubits which is "
                    "not in `state`: "
                    f"qubits in `circuit_append` = "
                    f"{circuit_append.all_qubits()}; "
                    f"`state` = {state}.")
        elif circuit_append is None:
            circuit_append = cirq.Circuit()
        else:
            raise TypeError(
                f"Invalid type for `circuit_append`: "
                f"{type(circuit_append)}"
            )

    else:  # state is None
        if circuit_prepend is None:
            if circuit_append is None:
                raise ValueError(
                    "`state`, `circuit_prepend` and `circuit.append` "
                    "cannot all be None at the same time."
                )
            else:
                state = circuit_append.all_qubits()
                circuit_prepend = cirq.Circuit()

        else:
            if circuit_append is None:
                state = circuit_prepend.all_qubits()
                circuit_append = cirq.Circuit()
            else:
                state = (circuit_prepend.all_qubits()
                         .union(circuit_append.all_qubits()))

    return state, circuit_prepend, circuit_append


class SWAPTestOutputLayer(SWAPTestLayer):
    def __init__(
            self,
            state1: typing.Union[cirq.Qid, typing.Tuple[cirq.GridQubit]],
            state2: typing.Union[cirq.Qid, typing.Tuple[cirq.GridQubit]],
            *,
            operators=None,
            circuit_prepend_of_state1: typing.Optional[cirq.Circuit] = None,
            circuit_prepend_of_state2: typing.Optional[cirq.Circuit] = None,
            circuit_append: typing.Optional[cirq.Circuit] = None,
            repetitions=None,
            backend=None,
            differentiator=None,
            initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
            regularizer=None,
            constraint=None,
            **kwargs
    ):
        """Instantiate this layer."""
        super().__init__(state1, state2, **kwargs)

        # Get all symbols in four circuits.
        def symbols_list_in_circuit(model_circuit) -> typing.List[str]:
            if model_circuit is None:  # empty circuit
                return []

            if not isinstance(model_circuit, cirq.Circuit):
                raise TypeError("model_circuit must be a cirq.Circuit object "
                                "or None. Given: {}".format(model_circuit))

            _symbols_list = list(
                sorted(tfq.util.get_circuit_symbols(model_circuit)))

            return _symbols_list

        _symbols_prepend_of_state1 = symbols_list_in_circuit(
            circuit_prepend_of_state1)
        _symbols_prepend_of_state2 = symbols_list_in_circuit(
            circuit_prepend_of_state2)
        _symbols_append = symbols_list_in_circuit(circuit_append)

        self._symbols_list = list(sorted(
            set.union(
                set(_symbols_prepend_of_state1),
                set(_symbols_prepend_of_state2),
                set(_symbols_append)
            )
        ))
        self._symbols = tf.constant([str(x) for x in self._symbols_list])

        if len(self._symbols_list) == 0:
            raise ValueError("model_circuit has no sympy.Symbols. Please "
                             "provide a circuit that contains symbols so "
                             "that their values can be trained.")

        if circuit_prepend_of_state1 is None:
            circuit_prepend_of_state1 = cirq.Circuit()
        # Make sure all qubits in state1 are included in circuit_prepend_of_state1
        # TODO: more elegant methods?
        for q in self.state1:
            if q not in circuit_prepend_of_state1.all_qubits():
                circuit_prepend_of_state1 += cirq.I(q)

        if circuit_prepend_of_state2 is None:
            circuit_prepend_of_state2 = cirq.Circuit()
        for q in self.state2:
            if q not in circuit_prepend_of_state2.all_qubits():
                circuit_prepend_of_state2 += cirq.I(q)

        if circuit_append is None:
            circuit_append = cirq.Circuit()

        (self._circuit_prepend_of_state1,
         self._circuit_prepend_of_state2,
         self._circuit_append) = tf.transpose(
            tfq.util.convert_to_tensor(
                [[circuit_prepend_of_state1,
                  circuit_prepend_of_state2,
                  circuit_append]]
            ), perm=[1, 0])

        # Ingest operators.
        if not operators:  # if `operators` is None or an empty iterable
            operators = [cirq.PauliString(cirq.Z(self.auxiliary_qubit))]
        else:
            if isinstance(operators, (cirq.PauliString, cirq.PauliSum)):
                operators = [operators]
            if not isinstance(operators, (list, np.ndarray, tuple)):
                raise TypeError("operators must be a cirq.PauliSum or "
                                "cirq.PauliString, or a list, tuple, "
                                "or np.array containing them. "
                                "Got {}.".format(type(operators)))
            if not all([
                isinstance(op, (cirq.PauliString, cirq.PauliSum))
                for op in operators
            ]):
                raise TypeError("Each element in operators to measure "
                                "must be a cirq.PauliString"
                                " or cirq.PauliSum")

            operators = np.array(
                [cirq.PauliString(cirq.Z(self.auxiliary_qubit))] + list(operators)
            )
        self._operators = tfq.util.convert_to_tensor([operators])

        # Ingest and promote repetitions.
        self._analytic = False
        if repetitions is None:
            self._analytic = True
        if not self._analytic and not isinstance(repetitions, numbers.Integral):
            raise TypeError("repetitions must be a positive integer value."
                            " Given: ".format(repetitions))
        if not self._analytic and repetitions <= 0:
            raise ValueError("Repetitions must be greater than zero.")
        if not self._analytic:
            self._repetitions = tf.constant(
                [[repetitions for _ in range(len(operators))]],
                dtype=tf.dtypes.int32)

        # Set backend and differentiator.
        if not isinstance(backend, cirq.Sampler
                          ) and repetitions is not None and backend is not None:
            raise TypeError("provided backend does not inherit cirq.Sampler "
                            "and repetitions!=None. Please provide a backend "
                            "that inherits cirq.Sampler or set "
                            "repetitions=None.")
        if not isinstance(backend, cirq.SimulatesFinalState
                          ) and repetitions is None and backend is not None:
            raise TypeError("provided backend does not inherit "
                            "cirq.SimulatesFinalState and repetitions=None. "
                            "Please provide a backend that inherits "
                            "cirq.SimulatesFinalState or choose a positive "
                            "number of repetitions.")
        if self._analytic:
            self._executor = tfq.layers.Expectation(
                backend=backend, differentiator=differentiator)
        else:
            self._executor = tfq.layers.SampledExpectation(
                backend=backend, differentiator=differentiator)

        self._append_layer = tfq.layers.AddCircuit()

        # Set additional parameter controls.
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)

        # Weight creation is not placed in a Build function because the number
        # of weights is independent of the input shape.
        self.parameters = self.add_weight('parameters',
                                          shape=self._symbols.shape,
                                          initializer=self.initializer,
                                          regularizer=self.regularizer,
                                          constraint=self.constraint,
                                          dtype=tf.float32,
                                          trainable=True)

    @property
    def symbols(self):
        """
        The symbols that are managed by this layer (in-order).

        Note:
            `symbols[i]` indicates what symbol name the managed variables in
            this layer map to.

        """
        return [sympy.Symbol(x) for x in self._symbols_list]

    def symbol_values(self):
        """
        Returns a Python `dict` containing symbol name, value pairs.

        :returns:
            Python `dict` with `str` keys and `float` values representing
            the current symbol values.
        """
        return dict(zip(self.symbols, self.get_weights()[0]))

    def call(self, inputs, add_metric=False):
        """Keras call function."""
        original_inputs = inputs
        inputs = tfq.append_circuit(inputs[0], inputs[1])
        circuit_batch_dim = tf.gather(tf.shape(inputs), 0)

        # SWAP test circuit:
        swap_test_circuit = cirq.Circuit()
        _, _ = add_swap_test(
            state1=self.state1,
            state2=self.state2,
            circuit=swap_test_circuit,
            auxiliary_qubit=self.auxiliary_qubit,
            cswap_in_elementary_gates=True
        )
        swap_test_circuit_tensor = tf.tile(
            tfq.convert_to_tensor([swap_test_circuit]), [circuit_batch_dim]
        )

        # Prepend circuit of state1:
        tiled_up_circuit_prepend_of_state1 = tf.tile(
            self._circuit_prepend_of_state1, [circuit_batch_dim]
        )
        model = tfq.layers.AddCircuit()(
            inputs, append=tiled_up_circuit_prepend_of_state1
        )

        # Prepend circuit of state2:
        tiled_up_circuit_prepend_of_state2 = tf.tile(
            self._circuit_prepend_of_state2, [circuit_batch_dim]
        )
        model = tfq.layers.AddCircuit()(
            model, append=tiled_up_circuit_prepend_of_state2
        )

        # Append SWAP test circuit
        model = tfq.layers.AddCircuit()(
            model, append=swap_test_circuit_tensor
        )

        # Append circuit:
        tiled_up_circuit_append = tf.tile(
            self._circuit_append, [circuit_batch_dim]
        )
        model = tfq.layers.AddCircuit()(
            model, append=tiled_up_circuit_append
        )

        # Tile up parameters and operators
        tiled_up_parameters = tf.tile([self.parameters], [circuit_batch_dim, 1])
        tiled_up_operators = tf.tile(self._operators, [circuit_batch_dim, 1])

        # Add metric (analytical inner product between two states)
        if add_metric:
            state1_tensor = tfq.layers.State()(
                tfq.layers.AddCircuit()(
                    original_inputs[0],
                    append=tiled_up_circuit_prepend_of_state1
                ),
                symbol_names=self._symbols,
                symbol_values=tiled_up_parameters
            ).to_tensor()
            state2_tensor = tfq.layers.State()(
                tfq.layers.AddCircuit()(
                    original_inputs[1],
                    append=tiled_up_circuit_prepend_of_state2
                ),
                symbol_names=self._symbols,
                symbol_values=tiled_up_parameters
            ).to_tensor()
            # print(state1_tensor)
            # print(state2_tensor)

            inner_products = tf.abs(
                tf.reduce_sum(
                    tf.multiply(tf.math.conj(state1_tensor), state2_tensor),
                    axis=1,
                    keepdims=True
                )
            )

            self.add_metric(
                inner_products,
                name="inner_product_abs",
                aggregation="mean"
            )

        # this is disabled to make autograph compilation easier.
        # pylint: disable=no-else-return
        if self._analytic:
            return self._executor(model,
                                  symbol_names=self._symbols,
                                  symbol_values=tiled_up_parameters,
                                  operators=tiled_up_operators)
        else:
            tiled_up_repetitions = tf.tile(self._repetitions,
                                           [circuit_batch_dim, 1])
            return self._executor(model,
                                  symbol_names=self._symbols,
                                  symbol_values=tiled_up_parameters,
                                  operators=tiled_up_operators,
                                  repetitions=tiled_up_repetitions)
        # pylint: enable=no-else-return
