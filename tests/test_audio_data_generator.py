import sys

sys.path.append("../")
import unittest
import tensorflow as tf
from audio_data_generator import AudioDataGenerator


class TestGenerator(unittest.TestCase):
    def test_flow_from_dir_dense(self):
        tr_gen = AudioDataGenerator()
        val_gen = AudioDataGenerator()

        tr_gen = tr_gen.flow_from_directory(
            "../test_audio/train",
            class_mode="binary",
            batch_size=32,
        )

        val_gen = val_gen.flow_from_directory(
            "../test_audio/validation",
            class_mode="binary",
            batch_size=32,
        )

        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.losses.binary_crossentropy,
            metrics=["acc"],
        )
        model.fit(
            tr_gen,
            epochs=3,
            validation_data=val_gen,
            validation_steps=10,
            steps_per_epoch=10,
        )
        print("=" * 70)

    def test_flow_from_dir_conv1d(self):
        gen = AudioDataGenerator()
        gen = gen.flow_from_directory(
            "../test_audio/train",
            class_mode="binary",
            batch_size=32,
            dim=2,
        )

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    10, kernel_size=3, strides=1, activation=tf.nn.relu
                ),
                # tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.binary_crossentropy
        )
        model.fit(gen, epochs=3)
        print("=" * 70)


if __name__ == "__main__":
    unittest.main()
