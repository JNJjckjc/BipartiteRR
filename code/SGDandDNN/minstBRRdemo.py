import numpy as np
import tensorflow as tf
import math
from absl import app, flags, logging

# 定义超参数
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('epsilon', 0.005, 'Privacy budget')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 2048, 'Batch size')
flags.DEFINE_integer('epochs', 10, 'Number of epochs')
FLAGS = flags.FLAGS



class AdaptiveClipping:
    def __init__(self, initial_C, decay_rate=0.99, min_C=1e-3):
        self.C = initial_C
        self.decay_rate = decay_rate
        self.running_mean = initial_C
        self.min_C = min_C

    def update(self, grad_norms):
        batch_mean_norm = np.mean(grad_norms)
        self.running_mean = self.decay_rate * self.running_mean + (1 - self.decay_rate) * batch_mean_norm
        self.C = max(self.running_mean, self.min_C)



def load_mnist():
    train, test = tf.keras.datasets.mnist.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(train_data, dtype=np.float32) / 255

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



loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)



def tinimoto_similarity(x, y):
    return abs(x - y)

def compute_similarity_matrix(sequence):
    n = len(sequence)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = tinimoto_similarity(sequence[i], sequence[j])

    return similarity_matrix

def sort_similarity(similarity_matrix):
    sorted_similarity = []
    sorted_index = []

    for row in similarity_matrix:
        sorted_indices = np.argsort(row)
        sorted_row = row[sorted_indices]
        sorted_similarity.append(sorted_row)
        sorted_index.append(sorted_indices)

    return sorted_similarity, sorted_index

def assign_weights(sorted_similarity, epsilon):
    weights = []

    for row in sorted_similarity:
        row_weights = [np.exp(epsilon)] + [1] * (len(row) - 1)
        weights.append(row_weights)

    return weights

def normalize_weights(weights):
    normalized_weights = []

    for row in weights:
        row_sum = sum(row)
        normalized_row = [w / row_sum for w in row]
        normalized_weights.append(normalized_row)

    return normalized_weights

def compute_partial_derivative(sorted_similarity, weights, i):
    partial_derivative = 0

    for j in range(len(weights[0])):
        grad_ij = 0
        for k in range(len(sorted_similarity)):
            grad_ij += (sorted_similarity[k][i] - sorted_similarity[k][j]) * weights[k][j]
        partial_derivative += grad_ij

    return partial_derivative

def calculate_item_probabilities(c, gap, n_divide, input_item, epsilon):
    items = [-c + i * gap for i in range(0, 2 * n_divide + 1)]
    similarity_matrix = compute_similarity_matrix(items)
    sorted_similarity, sorted_index = sort_similarity(similarity_matrix)
    weights = assign_weights(sorted_similarity, epsilon)
    normalized_weights = normalize_weights(weights)

    sorted_similarity, _ = sort_similarity(compute_similarity_matrix(items))

    m = 1
    for i in range(1, len(weights[0]) + 1):
        partial_derivative = compute_partial_derivative(sorted_similarity, weights, i - 1)
        if partial_derivative < 0:
            m = i
            for k in range(len(weights)):
                for j in range(1, m):
                    weights[k][j] = math.exp(epsilon)
        else:
            break

    normalized_weights = normalize_weights(weights)

    sequence_probabilities = []
    for i in range(len(items)):
        sorted_sequence_values = [items[idx] for idx in sorted_index[i]]
        probability_dict = {sorted_sequence_values[j]: normalized_weights[i][j] for j in range(len(items))}
        sequence_probabilities.append(probability_dict)

    closest_item_index = min(range(len(items)), key=lambda i: abs(items[i] - input_item))
    closest_sequence_probability = sequence_probabilities[closest_item_index]

    return closest_sequence_probability
    items, probs = zip(*probability_dict.items())
    selected_value = np.random.choice(items, p=probs)
    return selected_value


# 主函数
def main(argv):
    logging.set_verbosity(logging.INFO)
    train_data, train_labels, test_data, test_labels = load_mnist()
    model = create_model()

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
    adaptive_clipping = AdaptiveClipping(FLAGS.l2_norm_clip)

    for epoch in range(FLAGS.epochs):
        grad_norms_epoch = []

        for i in range(0, len(train_data), FLAGS.batch_size):
            batch_images = train_data[i:i + FLAGS.batch_size]
            batch_labels = train_labels[i:i + FLAGS.batch_size]

            with tf.GradientTape() as tape:
                logits = model(batch_images, training=True)
                loss = loss_fn(batch_labels, logits)

            gradients = tape.gradient(loss, model.trainable_variables)
            grad_norms_batch = [tf.norm(grad).numpy() for grad in gradients]
            grad_norms_epoch.extend(grad_norms_batch)

        adaptive_clipping.update(grad_norms_epoch)
        C = adaptive_clipping.C

        for i in range(0, len(train_data), FLAGS.batch_size):
            batch_images = train_data[i:i + FLAGS.batch_size]
            batch_labels = train_labels[i:i + FLAGS.batch_size]

            with tf.GradientTape() as tape:
                logits = model(batch_images, training=True)
                loss = loss_fn(batch_labels, logits)

            gradients = tape.gradient(loss, model.trainable_variables)
            clipped_gradients = []

            for grad in gradients:
                norm = tf.norm(grad)
                clipped_grad = grad / tf.maximum(1.0, norm / C)
                clipped_gradients.append(clipped_grad)

            noisy_gradients = []
            #sensitivity = C / FLAGS.batch_size
            for grad in clipped_gradients:
                grad_numpy = grad.numpy()
                for index, value in np.ndenumerate(grad_numpy):
                    grad_numpy[index]= calculate_item_probabilities(C, C/10, 10, value, FLAGS.epsilon)
                noisy_grad = tf.convert_to_tensor(grad_numpy, dtype=tf.float32)
                noisy_gradients.append(noisy_grad)

            optimizer.apply_gradients(zip(noisy_gradients, model.trainable_variables))

        loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    app.run(main)
