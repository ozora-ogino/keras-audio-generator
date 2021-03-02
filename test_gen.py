import unittest
import tensorflow as tf
from keras_preprocessing_audio import AudioDataGenerator


class TestGenerator(unittest.TestCase):
    def test_flow_from_dir(self):
        gen = AudioDataGenerator()
        gen = gen.flow_from_directory(
                'test_audio',
                class_mode='binary'
                batch_size=32,
                )

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
            ])

        model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.losses.binary_crossentropy)
        model.fit(gen, epochs=3, validation_split=0.2)

if __name__ == '__main__':
    unittest.main()
