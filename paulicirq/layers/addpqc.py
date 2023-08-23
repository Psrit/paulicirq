import cirq
import numpy as np
import sympy
import tensorflow as tf
from tensorflow_quantum.core.ops import tfq_utility_ops
from tensorflow_quantum.python import util


class AddPQC(tf.keras.layers.Layer):
    def __init__(
        self,
        model_circuit,
        *,
        initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
        regularizer=None,
        constraint=None,
        **kwargs,
    ):
        """
        Instantiate this layer.

        Create a layer that will append/prepend a given (parameterized) circuit
        to the inputs. This layer will accept one input tensor representing a
        quantum data source (these circuits may contain some symbols) and
        append/prepend the model_circuit to them, and then output the assembled

        For example, PQCs can be connected as follows:

        ```python
        class MyModel(tf.keras.models.Model):
            def __init__(self):
                super().__init__(self)
                self.add_layer1: AddPQC = AddPQC(
                    pqc1,
                    constraint=tf.keras.constraints.MinMaxNorm(
                        min_value=theta_bounds[0], max_value=theta_bounds[1]
                    ),
                    initializer=tf.keras.initializers.RandomUniform(
                        theta_bounds[0], theta_bounds[1]
                    )
                )
                self.add_layer2: AddPQC = AddPQC(
                    pqc2,
                    constraint=tf.keras.constraints.MinMaxNorm(
                        min_value=phi_bounds[0], max_value=phi_bounds[1]
                    ),
                    initializer=tf.keras.initializers.RandomUniform(
                        phi_bounds[0], phi_bounds[1]
                    )
                )
                self.expectation_layer = tfq.layers.Expectation(
                    differentiator=tfq.differentiators.ParameterShift()
                )

            def call(self, inputs):
                x = self.add_layer1(
                    inputs, append=True
                )
                x = self.add_layer2(
                    x, append=True
                )

                symbol_values = tf.concat([
                    self.add_layer1.get_parameters(),
                    self.add_layer2.get_parameters()
                ], axis=1)
                outputs = self.expectation_layer(
                    x,
                    operators=[cirq.Z(qubit)],
                    symbol_names=[theta, phi],
                    symbol_values=tf.tile(
                        symbol_values,
                        [tf.shape(inputs)[0], 1]
                    )
                )

                return outputs
        ```

        Important note: ONLY use this class inside a customized Model/Layer's
        `call` method with the expectation value of PQC calculated (i.e. don't
        use it to assemble a Sequential model).
        This is because the `call` method of `tfq.layers.Expectation` (which
        is necessary for a measurement-based quantum model) requires a tiled
        `symbol_values` whose first dimension length equals the batch size,
        which should be determined until the training stage (unless it is
        hard-coded).

        model_circuit:
            `cirq.Circuit` containing `sympy.Symbols` that will be used as the
            model which will be fed quantum data inputs.
            If it is parameterized, the symbols it contains will be used as the
            trainable weights of this layer.
        backend: (Optional)
            Backend to use to simulate states. Defaults to the native TensorFlow
            simulator (None).
            However users may also specify a preconfigured cirq simulation
            object to use instead. If a cirq object is given it must inherit
            either `cirq.SimulatesFinalState` if analytic expectations are
            desired or `cirq.Sampler` if sampled expectations are desired.
        differentiator: (Optional)
            `tfq.differentiator` object to specify how gradients of
            `model_circuit` should be calculated.
        initializer: (Optional)
            `tf.keras.initializer` object to specify how the symbols in
            `model_circuit` should be initialized when creating the managed
            variables.
        regularizer: (Optional)
            `tf.keras.regularizer` object applied to the managed variables when
            parameterizing `model_circuit`.
        constraint: (Optional)
            `tf.keras.constraint` object applied to the managed variables when
            parameterizing `model_circuit`.

        """
        super().__init__(**kwargs)

        # Ingest model_circuit.
        if not isinstance(model_circuit, cirq.Circuit):
            raise TypeError("model_circuit must be a cirq.Circuit object."
                            " Given: {}".format(model_circuit))

        # Get circuit symbols
        self._symbols_list = list(
            sorted(util.get_circuit_symbols(model_circuit))
        )
        self._symbols = tf.constant([str(x) for x in self._symbols_list])

        # Convert `model_circuit` into tf.Tensor
        self._model_circuit = util.convert_to_tensor([model_circuit])

        # Set additional parameter controls.
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)

        # # Weight creation is not placed in a Build function because the number
        # # of weights is independent of the input shape.
        self.parameters = self.add_weight('parameters',
                                          shape=self._symbols.shape,
                                          initializer=self.initializer,
                                          regularizer=self.regularizer,
                                          constraint=self.constraint,
                                          dtype=tf.float32,
                                          trainable=True)

    def build(self, input_shape):
        """Keras build function."""
        super().build(input_shape)

    def call(self, inputs, *, append=False, prepend=False):
        """
        Keras call method.

        Input options:

        `inputs`:
            Can be a single `cirq.Circuit`, a Python `list` of `cirq.Circuit`s
            or a pre-converted `tf.Tensor` of `cirq.Circuit`s.

        `append`:
            A boolean specifying whether the parameterized model circuit will be
            appended. `append` and `prepend` are mutually exclusive, i.e. one
            and only one of them is True.

        `prepend`:
            A boolean specifying whether the parameterized model circuit will be
            prepended. `append` and `prepend` are mutually exclusive, i.e. one
            and only one of them is True.

        Output shape:
            `tf.Tensor` of shape [input size] containing circuits with append
            circuits appended or prepend circuits prepended.

        """
        # inputs and outputs are circuits.

        if append is False and prepend is False:
            raise ValueError(
                "The model circuit must be either appended or prepended."
            )

        if append is True and prepend is True:
            raise ValueError(
                "The model circuit cannot be both appended and prepended."
            )

        # Ingest input circuit(s).
        if isinstance(inputs, cirq.Circuit):
            inputs = util.convert_to_tensor([inputs])

        if isinstance(inputs, (tuple, list, np.ndarray)):
            inputs = util.convert_to_tensor(inputs)

        if not tf.is_tensor(inputs):
            raise TypeError(
                "Circuits cannot be parsed with given input: {}".format(inputs)
            )

        batch_dim = tf.gather(tf.shape(inputs), 0)
        model_circuit = tf.tile(self._model_circuit, [batch_dim])

        # Append circuit:
        if append is True:
            return tfq_utility_ops.append_circuit(inputs, model_circuit)

        # Otherwise prepend circuit.
        else:
            return tfq_utility_ops.append_circuit(model_circuit, inputs)

    @property
    def symbols(self):
        """
        The symbols that are managed by this layer (in-order).

        Note: `symbols[i]` indicates what symbol name the managed variables in
        this layer map to.

        """
        return [sympy.Symbol(x) for x in self._symbols_list]

    def symbol_values(self):
        """
        Returns a Python `dict` containing symbol name, value pairs.

        Returns:
            Python `dict` with `str` keys and `float` values representing
            the current symbol values.

        """
        return dict(zip(self.symbols, self.get_weights()[0]))

    def get_parameters(self, circuit_batch_dim=1):
        return tf.tile([self.parameters], [circuit_batch_dim, 1])
