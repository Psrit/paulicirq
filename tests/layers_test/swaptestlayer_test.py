import unittest

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow_quantum.core.ops import tfq_utility_ops

from paulicirq.gates.general_rotation import General1BitRotation
from paulicirq.layers.swaptest import SWAPTestLayer, SWAPTestOutputLayer
from paulicirq.utils import generate_random_rotation_batch, act_gate_batch_on_qubits
from paulicirq.utils import tf_allclose


class SWAPTestLayerTest(unittest.TestCase):
    def setUp(self):
        self.num_qubits_of_a_state = 2
        self.state1 = [cirq.GridQubit(0, i) for i in range(self.num_qubits_of_a_state)]
        self.state2 = [cirq.GridQubit(1, i) for i in range(self.num_qubits_of_a_state)]
        self.circuit_batch_size = 5

    def test_circuit(self):
        circuit_input1 = tfq.convert_to_tensor(
            [
                cirq.Circuit([
                    cirq.X.on_each(*self.state1)
                ])
            ] * 2
        )  # NECESSARY to *2!
        circuit_input2 = tfq.convert_to_tensor([
            cirq.Circuit([
                cirq.X.on_each(*self.state2)
            ]),
            cirq.Circuit(
                cirq.I.on_each(*self.state2)
            )
        ])

        output: tf.Tensor = SWAPTestLayer(self.state1, self.state2)(
            [circuit_input1, circuit_input2]
        )
        # print(output)
        print(tfq.from_tensor(output))
        self.assertTrue(
            output.shape == (2,)
        )

    def test_swap_overlap_identical(self):
        random_rotations = generate_random_rotation_batch(
            self.num_qubits_of_a_state,
            self.circuit_batch_size
        )

        circuit_input1 = tfq.convert_to_tensor(
            act_gate_batch_on_qubits(
                random_rotations, self.state1,
                as_circuits=True
            )
        )
        circuit_input2 = tfq.convert_to_tensor(
            act_gate_batch_on_qubits(
                random_rotations, self.state2,
                as_circuits=True
            )
        )

        swap_layer: SWAPTestLayer = SWAPTestLayer(
            self.state1, self.state2
        )
        swap_test: tf.Tensor = swap_layer(
            [circuit_input1, circuit_input2]
        )

        output: tf.Tensor = tfq.layers.Expectation()(
            swap_test,
            operators=[cirq.Z(swap_layer.auxiliary_qubit)]
        )
        print(output)
        self.assertTrue(
            tf_allclose(
                output,
                tf.convert_to_tensor([[1.0]] * self.circuit_batch_size),
                rtol=1e-2
            )
        )

    def test_swap_overlap_orthogonal(self):
        random_rotations = generate_random_rotation_batch(
            self.num_qubits_of_a_state,
            self.circuit_batch_size
        )

        circuit_input1 = tfq.convert_to_tensor(
            act_gate_batch_on_qubits(
                random_rotations, self.state1,
                as_circuits=True
            )
        )
        circuit_input2 = tfq.convert_to_tensor(
            act_gate_batch_on_qubits(
                random_rotations, self.state2,
                as_circuits=True
            )
        )
        circuit_input2 = tfq_utility_ops.tfq_append_circuit(
            tfq.convert_to_tensor([
                                      cirq.Circuit([cirq.X.on_each(self.state2)])
                                  ] * self.circuit_batch_size),
            circuit_input2
        )

        swap_layer: SWAPTestLayer = SWAPTestLayer(
            self.state1, self.state2
        )
        swap_test: tf.Tensor = swap_layer(
            [circuit_input1, circuit_input2]
        )

        output: tf.Tensor = tfq.layers.Expectation()(
            swap_test,
            operators=[cirq.Z(swap_layer.auxiliary_qubit)]
        )
        print(output)
        self.assertTrue(
            tf_allclose(
                output,
                tf.convert_to_tensor([[0.0]] * self.circuit_batch_size),
                rtol=1e-5,
                atol=1e-5
            )
        )

    def test_swap_overlap_random_rotations(self):
        circuit_input1 = tfq.convert_to_tensor(
            act_gate_batch_on_qubits(
                generate_random_rotation_batch(
                    self.num_qubits_of_a_state,
                    self.circuit_batch_size
                ),
                self.state1,
                as_circuits=True
            )
        )
        circuit_input2 = tfq.convert_to_tensor(
            [cirq.Circuit(
                [cirq.I.on_each(*self.state2)]
            )] * self.circuit_batch_size
        )

        swap_layer: SWAPTestLayer = SWAPTestLayer(
            self.state1, self.state2
        )
        swap_test: tf.Tensor = swap_layer(
            [circuit_input1, circuit_input2]
        )

        output: tf.Tensor = tfq.layers.Expectation()(
            swap_test,
            operators=[cirq.Z(swap_layer.auxiliary_qubit)]
        )
        print(output)


class SWAPTestOutputLayerTest(unittest.TestCase):
    def setUp(self):
        self.num_qubits_of_a_state = 2
        self.state1 = tuple(
            cirq.GridQubit(0, i) for i in range(self.num_qubits_of_a_state)
        )
        self.state2 = tuple(
            cirq.GridQubit(1, i) for i in range(self.num_qubits_of_a_state)
        )
        self.circuit_batch_size = 5

    def test_circuit(self):
        circuit_input1 = tfq.convert_to_tensor(
            [
                cirq.Circuit([
                    cirq.X.on_each(*self.state1)
                ])
            ] * 2
        )  # NECESSARY to *2!
        circuit_input2 = tfq.convert_to_tensor([
            cirq.Circuit([
                cirq.X.on_each(*self.state2)
            ]),
            cirq.Circuit(
                cirq.I.on_each(*self.state2)
            )
        ])

        output: tf.Tensor = SWAPTestOutputLayer(
            operators=[
                cirq.PauliString(
                    cirq.Z(self.state1[0]),
                    cirq.Z(self.state2[0])
                ),
                cirq.PauliString(
                    cirq.Z(self.state1[1]),
                    cirq.Z(self.state2[1])
                )
            ],
            circuit_prepend_of_state1=cirq.Circuit(
                [cirq.rx(sympy.Symbol("theta1"))(self.state1[0])]
            ),
            circuit_prepend_of_state2=cirq.Circuit(
                [cirq.rx(sympy.Symbol("theta2"))(self.state2[0])]
            ),
            circuit_append=cirq.Circuit(
                [cirq.rx(-sympy.Symbol("theta1"))(self.state1[0]),
                 cirq.rx(-sympy.Symbol("theta2"))(self.state2[0])]
            ),
            state1=self.state1, state2=self.state2
        )(
            [circuit_input1, circuit_input2]
        )
        print(output)
        self.assertTrue(
            output.shape == (2, 2 + 1)
        )

    def test_swap_overlap_identical(self):
        random_rotations = generate_random_rotation_batch(
            self.num_qubits_of_a_state,
            self.circuit_batch_size
        )

        circuit_input1 = tfq.convert_to_tensor(
            act_gate_batch_on_qubits(
                random_rotations, self.state1,
                as_circuits=True
            )
        )
        circuit_input2 = tfq.convert_to_tensor(
            act_gate_batch_on_qubits(
                random_rotations, self.state2,
                as_circuits=True
            )
        )

        swap_layer: SWAPTestOutputLayer = SWAPTestOutputLayer(
            circuit_prepend_of_state1=cirq.Circuit(
                [cirq.rx(sympy.Symbol("theta"))(self.state1[0])]
            ),
            circuit_prepend_of_state2=cirq.Circuit(
                [cirq.rx(sympy.Symbol("theta"))(self.state2[0])]
            ),
            state1=self.state1, state2=self.state2
        )
        output: tf.Tensor = swap_layer(
            [circuit_input1, circuit_input2]
        )

        print(output)
        self.assertTrue(
            tf_allclose(
                output,
                tf.convert_to_tensor([[1.0]] * self.circuit_batch_size),
                rtol=1e-2
            )
        )

    def test_swap_overlap_orthogonal(self):
        random_rotations = [General1BitRotation(
            sympy.Symbol(f"theta_q{i}_1"),
            sympy.Symbol(f"theta_q{i}_2"),
            sympy.Symbol(f"theta_q{i}_3")
        ) for i in range(self.num_qubits_of_a_state)]

        circuit_input1 = tfq.convert_to_tensor(
            [cirq.Circuit()] * self.circuit_batch_size
        )
        circuit_input2 = tfq.convert_to_tensor(
            [cirq.Circuit(cirq.X.on_each(self.state2))]
            * self.circuit_batch_size
        )

        swap_layer = SWAPTestOutputLayer(
            circuit_prepend_of_state1=act_gate_batch_on_qubits(
                [random_rotations], self.state1, as_circuits=True
            )[0],
            circuit_prepend_of_state2=act_gate_batch_on_qubits(
                [random_rotations], self.state2, as_circuits=True
            )[0],
            state1=self.state1, state2=self.state2
        )
        output: tf.Tensor = swap_layer(
            [circuit_input1, circuit_input2]
        )
        print(output)
        self.assertTrue(
            tf_allclose(
                output,
                tf.convert_to_tensor([[0.0]] * self.circuit_batch_size),
                rtol=1e-5,
                atol=1e-5
            )
        )

    def test_swap_overlap_random_rotations(self):
        circuit_input1 = tfq.convert_to_tensor(
            act_gate_batch_on_qubits(
                generate_random_rotation_batch(
                    self.num_qubits_of_a_state,
                    self.circuit_batch_size
                ),
                self.state1,
                as_circuits=True
            )
        )
        circuit_input2 = tfq.convert_to_tensor(
            [cirq.Circuit()] * self.circuit_batch_size
        )

        swap_layer = SWAPTestOutputLayer(
            circuit_prepend_of_state1=cirq.Circuit(
                [cirq.rx(sympy.Symbol("theta1"))(self.state1[0])]
            ),
            circuit_prepend_of_state2=cirq.Circuit(
                [cirq.rx(sympy.Symbol("theta2"))(self.state2[0])]
            ),
            circuit_append=cirq.Circuit(
                [cirq.rx(-sympy.Symbol("theta1"))(self.state1[0]),
                 cirq.rx(-sympy.Symbol("theta2"))(self.state2[0])]
            ),
            state1=self.state1,
            state2=self.state2
        )
        output: tf.Tensor = swap_layer(
            [circuit_input1, circuit_input2]
        )
        print(output)
        self.assertEqual(
            output.shape, (5, 1)
        )

    def test_model(self):
        circuit_input1 = tfq.convert_to_tensor(
            [
                cirq.Circuit([
                    cirq.X.on_each(*self.state1)
                ])
            ] * 2
        )  # NECESSARY to *2!
        circuit_input2 = tfq.convert_to_tensor([
            cirq.Circuit([
                cirq.X.on_each(*self.state2)
            ]),
            cirq.Circuit(
                cirq.I.on_each(*self.state2)
            )
        ])

        input1_layer = tf.keras.layers.Input(shape=(), dtype=tf.string)
        input2_layer = tf.keras.layers.Input(shape=(), dtype=tf.string)
        output_layer: tf.Tensor = SWAPTestOutputLayer(
            operators=[
                cirq.PauliString(
                    cirq.Z(self.state1[0]),
                    cirq.Z(self.state2[0])
                ),
                cirq.PauliString(
                    cirq.Z(self.state1[1]),
                    cirq.Z(self.state2[1])
                )
            ],
            circuit_prepend_of_state1=cirq.Circuit(
                [cirq.rx(sympy.Symbol("theta1"))(self.state1[0])]
            ),
            circuit_prepend_of_state2=cirq.Circuit(
                [cirq.rx(sympy.Symbol("theta2"))(self.state2[0])]
            ),
            circuit_append=cirq.Circuit(
                [cirq.rx(-sympy.Symbol("theta1"))(self.state1[0]),
                 cirq.rx(-sympy.Symbol("theta2"))(self.state2[0])]
            ),
            state1=self.state1, state2=self.state2
        )(
            [input1_layer, input2_layer]
        )

        model = tf.keras.models.Model(
            inputs=[input1_layer, input2_layer],
            outputs=output_layer
        )
        model.summary()

        output_tensor = model([circuit_input1, circuit_input2])
        print(output_tensor)
        self.assertTrue(
            output_tensor.shape == (2, 2 + 1)
        )

    def test_empty_circuit(self):
        circuit_input1 = tfq.convert_to_tensor(
            [cirq.Circuit()]
        )
        circuit_input2 = tfq.convert_to_tensor(
            [cirq.Circuit()]
        )
        theta = sympy.Symbol("theta")

        input1_layer = tf.keras.layers.Input(shape=(), dtype=tf.string)
        input2_layer = tf.keras.layers.Input(shape=(), dtype=tf.string)

        swap_layer: SWAPTestOutputLayer = SWAPTestOutputLayer(
            circuit_prepend_of_state1=cirq.Circuit(
                [cirq.rx(theta)(self.state1[0]),
                 cirq.ry(theta)(self.state1[1])]
            ),
            circuit_prepend_of_state2=cirq.Circuit(
                [cirq.rx(theta)(self.state2[0]),
                 cirq.ry(theta)(self.state2[1])]
            ),
            state1=self.state1, state2=self.state2
        )
        swap_layer.set_weights([tf.convert_to_tensor([0.0])])
        output_layer: tf.Tensor = swap_layer(
            [input1_layer, input2_layer],
            add_metric=True
        )

        model = tf.keras.models.Model(
            inputs=[input1_layer, input2_layer],
            outputs=output_layer
        )
        model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError()
        )

        history = model.evaluate(
            x=[circuit_input1, circuit_input2],
            y=tf.convert_to_tensor([[1.0]])
        )
        print(history)

