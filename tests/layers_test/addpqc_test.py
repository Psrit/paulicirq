import unittest

import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from paulicirq.layers.addpqc import AddPQC

qubit = cirq.GridQubit(0, 0)


class AddPQCTest(unittest.TestCase):
    def setUp(self) -> None:
        # Input data (tensor of quantum circuit)
        self.input_circuit_tensors = tfq.convert_to_tensor([
            cirq.Circuit(cirq.X(qubit)),
            cirq.Circuit(cirq.Y(qubit)),
            cirq.Circuit(cirq.H(qubit))
        ])

        # PQC to be added
        self.theta = sympy.Symbol("theta")
        self.parameterized_circuit = cirq.Circuit(
            (cirq.X ** self.theta).on(qubit)
        )

        # AddPQC layer
        self.add_layer: AddPQC = AddPQC(
            self.parameterized_circuit,
            constraint=tf.keras.constraints.MinMaxNorm(
                min_value=0, max_value=2
            )
        )
        # added_tensor_0 = add_layer(
        #     input_circuit_tensors, append=True
        # )
        # added_tensor = add_layer(
        #     added_tensor_0, append=True
        # )

        self.expectation_layer = tfq.layers.Expectation(
            differentiator=tfq.differentiators.ForwardDifference(
                grid_spacing=0.0001
            )
        )

    def test_tensor_run(self):
        added_tensor = self.add_layer(
            self.input_circuit_tensors, append=True
        )

        # print(added_tensor)
        print(tfq.from_tensor(added_tensor))

        # print(self.add_layer.get_weights())
        # print(self.add_layer.symbol_values())

        self.add_layer.set_weights(np.array([[1]]))  # won't influence added_tensor, which is still parameterized

        theta_values = tf.convert_to_tensor([
            [1], [1], [1]
        ], dtype=tf.float32)  # the first dimension length must be equal to the batch size

        with tf.GradientTape() as g:
            g.watch(theta_values)
            output_tensor = self.expectation_layer(
                added_tensor,
                operators=[cirq.Z(qubit)],
                symbol_names=[self.theta],
                symbol_values=theta_values,
                # initializer=tf.keras.initializers.Zeros(),  # 若不使用 GradientTape，则可以通过 initializer 来初始化训练参数
            )

        # For t = 1:
        # <0| X X^t Z X^t X |0> = <0| I Z I |0> = 1
        # <0| Y X^t Z X^t Y |0> = (-i) i <0| Z Z Z |0> = <0| Z |0> = 1
        # <0| H X^t Z X^t H |0> = -<0|X|0> = 0
        self.assertTrue(np.allclose(
            output_tensor.numpy(),
            np.array(
                [[1],
                 [1],
                 [0]]
            )
        ))

        # ∂<0| A X^t Z X^t A |0>/∂t
        # = t [<0| A X^{t-1} Z X^t A |0> + c.c.]
        # = <0| A Z X A |0> + c.c.  (for t = 1)
        #   ┌   <0| X Z |0>   ┐   ┌ 0 ┐
        # = ⎪ <0| Y Z X Y |0> │ = │ 0 │
        #   └ <0| H Z X H |0> ┘   └ 0 ┘
        gradient_to_theta = g.gradient(output_tensor, theta_values)
        self.assertTrue(np.allclose(
            gradient_to_theta.numpy(),
            np.array(
                [[0],
                 [0],
                 [0]]
            )
        ))

    def test_model_predict(self):
        input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string)
        x = self.add_layer(
            input_layer, append=True
        )

        self.add_layer.set_weights(np.array([[1]]))  # 不会对 added_tensor 产生影响（仍然是含参数的）

        theta_values = tf.convert_to_tensor([
            [1], [1], [1]
        ], dtype=tf.float32)

        output_layer = self.expectation_layer(
            x,
            operators=[cirq.Z(qubit)],
            symbol_names=[self.theta],
            symbol_values=theta_values,
        )

        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer
        )

        output_tensor = model.predict(self.input_circuit_tensors)

        # For t = 1:
        # <0| X X^t Z X^t X |0> = <0| I Z I |0> = 1
        # <0| Y X^t Z X^t Y |0> = (-i) i <0| Z Z Z |0> = <0| Z |0> = 1
        # <0| H X^t Z X^t H |0> = -<0|X|0> = 0
        self.assertTrue(np.allclose(
            output_tensor,
            np.array(
                [[1],
                 [1],
                 [0]]
            )
        ))

    def test_model_fit(self):
        qubit = cirq.GridQubit(0, 0)

        # The first PQC to be added
        theta = sympy.Symbol("theta")
        pqc1 = cirq.Circuit(
            (cirq.X ** theta).on(qubit)
        )
        theta_bounds = (0, 2)

        # The second PQC to be added
        phi = sympy.Symbol("phi")
        pqc2 = cirq.Circuit(
            (cirq.Y ** phi).on(qubit)
        )
        phi_bounds = (0, 2)

        class TestModel(tf.keras.models.Model):
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

        model = TestModel()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
            loss=tf.keras.losses.MeanSquaredError()
        )

        print(model.variables)

        input_circuit = cirq.Circuit(cirq.X(qubit))

        batch_size = 5
        input_tensor = tfq.convert_to_tensor(
            [
                input_circuit
            ] * batch_size
        )
        y_tensor = tf.convert_to_tensor(
            [
                [1]
            ] * batch_size
        )

        # model(input_tensor)
        # print(model.summary())

        history = model.fit(
            input_tensor,
            y_tensor,
            batch_size,
            epochs=20,
            verbose=1
        )
        print(model.variables)
        print(model(tfq.convert_to_tensor([input_circuit])))  # ~1

        resolved_pqc = cirq.resolve_parameters(
            tfq.from_tensor(model.add_layer1._model_circuit)[0]
            + tfq.from_tensor(model.add_layer2._model_circuit)[0],
            param_resolver={
                theta: model.add_layer1.get_parameters().numpy()[0, 0],
                phi: model.add_layer2.get_parameters().numpy()[0, 0]
            }
        )
        z_probs = np.round(
            np.abs(
                cirq.unitary(input_circuit + resolved_pqc) @ np.array([1, 0])
            ) ** 2, 2
        )
        print(z_probs)
        # self.assertTrue(
        #     np.allclose(
        #         z_probs,  # simulated output probability distribution of <Z>
        #         np.array([1, 0]),
        #         atol=0.05,
        #         rtol=0.01
        #     )
        # )
