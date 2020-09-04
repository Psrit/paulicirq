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
        # 输入线路数据（张量）
        self.input_circuit_tensors = tfq.convert_to_tensor([
            cirq.Circuit(cirq.X(qubit)),
            cirq.Circuit(cirq.Y(qubit)),
            cirq.Circuit(cirq.H(qubit))
        ])

        # 需要添加的参数化线路
        self.theta = sympy.Symbol("theta")
        self.parameterized_circuit = cirq.Circuit(
            (cirq.X ** self.theta).on(qubit)
        )

        # AddPQC 层
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

        self.add_layer.set_weights(np.array([[1]]))  # 不会对 added_tensor 产生影响（仍然是含参数的）

        theta_values = tf.convert_to_tensor([
            [1], [1], [1]
        ], dtype=tf.float32)

        with tf.GradientTape() as g:
            g.watch(theta_values)
            output_tensor = self.expectation_layer(
                added_tensor,
                operators=[cirq.Z(qubit)],
                symbol_names=[self.theta],
                symbol_values=theta_values,
                # [[0]] or [[0], [0]]: exit code 139, SIGSEGV 11. 和 Expectation 的 call 规则有关???
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
        batch_size = 5
        input_tensor = tfq.convert_to_tensor(
            [
                cirq.Circuit(cirq.X(qubit))
            ] * batch_size
        )
        y_tensor = tf.convert_to_tensor(
            [
                [1]
            ] * batch_size
        )

        input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string)
        x = self.add_layer(
            input_layer, append=True
        )
        print(self.add_layer.parameters)
        output_layer = self.expectation_layer(
            x,
            operators=[cirq.Z(qubit)],
            symbol_names=[self.theta],
            symbol_values=self.add_layer.get_parameters()
        )
        # output_layer = tfq.layers.PQC(
        #     self.parameterized_circuit,
        #     operators=[cirq.Z(qubit)],
        #     constraint=tf.keras.constraints.MinMaxNorm(
        #         min_value=0, max_value=2
        #     )
        # )(input_layer)

        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
            loss=tf.keras.losses.MeanSquaredError()
        )

        print(model.get_weights())
        print(model.summary())

        history = model.fit(
            input_tensor,
            y_tensor,
            batch_size,
            epochs=20,
            verbose=1
        )
        print(model.get_weights())
