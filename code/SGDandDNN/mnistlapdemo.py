from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from absl import app, flags, logging

flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('epsilon', 0.0005, 'Privacy budget')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 2048, 'Batch size')
flags.DEFINE_integer('epochs', 200, 'Number of epochs')

FLAGS = flags.FLAGS


def add_laplace_noise(gradients, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=gradients.shape)
    return gradients + noise



def load_mnist():
    train, test = tf.keras.datasets.mnist.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    return train_data, train_labels, test_data, test_labels



def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Conv2D(32, 4, strides=2, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model



def per_example_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    return loss_fn(y_true, y_pred)



def main(unused_argv):
    logging.set_verbosity(logging.INFO)

    train_data, train_labels, test_data, test_labels = load_mnist()

    model = create_model()

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)


    for epoch in range(FLAGS.epochs):
        for i in range(0, len(train_data), FLAGS.batch_size):
            batch_images = train_data[i:i + FLAGS.batch_size]
            batch_labels = train_labels[i:i + FLAGS.batch_size]

            with tf.GradientTape() as tape:
                logits = model(batch_images, training=True)
                loss = per_example_loss(batch_labels, logits)

            gradients = tape.gradient(loss, model.trainable_variables)  # 这里是利用一个batch进行训练，然后求梯度
            clipped_gradients = []

            for grad in gradients:
                norm = tf.norm(grad)
                clipped_grad = grad / tf.maximum(1.0, norm / FLAGS.l2_norm_clip)
                clipped_gradients.append(clipped_grad)
            # 添加噪声
            sensitivity = FLAGS.l2_norm_clip / FLAGS.batch_size
            noisy_gradients = [add_laplace_noise(grad.numpy(), sensitivity, FLAGS.epsilon) for grad in
                               clipped_gradients]
            noisy_gradients = [tf.convert_to_tensor(grad, dtype=tf.float32) for grad in noisy_gradients]


            optimizer.apply_gradients(zip(noisy_gradients, model.trainable_variables))


        loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    app.run(main)
