#### 1. Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np
import pathlib

ep = 10 # Number of epochs

### 2. Dataset Preparation

#### MNIST Dataset
# Load and preprocess MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Expand dimensions to match the input shape (batch, height, width, channels)
x_train = np.expand_dims(x_train, -1)  # Shape becomes (batch_size, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)    # Shape becomes (batch_size, 28, 28, 1)

# Visualize some MNIST images
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i].squeeze(), cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

# Path to your local Caltech-101 dataset
caltech101_path = "/Users/soumalyamondal/Desktop/Project/Data/caltech-101/101_ObjectCategories"

# Convert the dataset to a TensorFlow Dataset
def load_local_dataset(data_dir, image_size=(128, 128), batch_size=32):
    data_dir = pathlib.Path(data_dir)
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=image_size,  # Resize images to 128x128
        batch_size=batch_size,
        shuffle=True  # Ensure data is shuffled
    )
    return dataset


# Load the full dataset (this includes both train and test)
full_dataset = load_local_dataset(caltech101_path)

# Calculate the size of train and test splits
train_size = int(0.8 * len(full_dataset))  # 80% for training
test_size = len(full_dataset) - train_size  # 20% for testing

# Split the dataset into training and test sets
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)



# Visualize some samples from the dataset
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"Label: {labels[i].numpy()}")
        plt.axis('off')
plt.show()


### 3. CNN Architecture and Training

# Model creation function with flexibility to change pooling type, loss, and optimizer
def create_cnn_model(input_shape, num_classes, pooling_type='max', loss_function='categorical_crossentropy', optimizer_name='adam'):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', name="conv_1")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', name="conv_2")(x)

    if pooling_type == 'max':
        x = layers.MaxPooling2D((2, 2))(x)
    else:
        x = layers.AveragePooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Choose optimizer
    if optimizer_name == 'adam':
        optimizer = optimizers.Adam()
    elif optimizer_name == 'sgd':
        optimizer = optimizers.SGD()
    elif optimizer_name == 'rmsprop':
        optimizer = optimizers.RMSprop()

    # Use SparseTopKCategoricalAccuracy if sparse_categorical_crossentropy is used
    if loss_function == 'sparse_categorical_crossentropy':
        top_k_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    else:
        top_k_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=5)

    # Create and compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', top_k_metric])  # Top-1 and Top-5 accuracy
    return model

### 4. Correct Encoding of Labels Based on Loss Function
def encode_labels(loss_function, labels):
    if loss_function == 'categorical_crossentropy':
        return tf.keras.utils.to_categorical(labels)
    return labels  # Keep integer labels for sparse_categorical_crossentropy

### 5. Training and Evaluation

# Function to train and evaluate models
def train_and_evaluate_model(model, train_data, test_data, epochs=2):
    history = model.fit(train_data[0], train_data[1], epochs=epochs, validation_data=test_data)

    # Evaluate on test data
    test_loss, test_acc, test_top_5_acc = model.evaluate(test_data[0], test_data[1])
    print(f"Test accuracy (Top 1): {test_acc}")
    print(f"Test accuracy (Top 5): {test_top_5_acc}")
    return history

### Train and Evaluate on MNIST with Pooling Options

# Train with Max Pooling and Average Pooling
def train_with_pooling_options(pooling_type):
    y_train_encoded = encode_labels('categorical_crossentropy', y_train)
    y_test_encoded = encode_labels('categorical_crossentropy', y_test)

    cnn_model_mnist = create_cnn_model(input_shape=(28, 28, 1), num_classes=10, pooling_type=pooling_type)
    history = train_and_evaluate_model(cnn_model_mnist, (x_train, y_train_encoded), (x_test, y_test_encoded), epochs=ep)

    return cnn_model_mnist, history

print("Training with Max Pooling on MNIST:")
cnn_model_mnist_max, history_max_mnist = train_with_pooling_options('max')

print("Training with Average Pooling on MNIST:")
cnn_model_mnist_avg, history_avg_mnist = train_with_pooling_options('avg')

### Train and Evaluate on Caltech-101 with Pooling Options

def train_and_evaluate_caltech_model(model, train_dataset, test_dataset, epochs=2):
    history_caltech = model.fit(train_dataset, epochs=ep, validation_data=test_dataset)

    # Evaluate on test data
    test_loss_caltech, test_acc_caltech, test_top_5_acc_caltech = model.evaluate(test_dataset)
    print(f"Caltech-101 Test accuracy (Top 1): {test_acc_caltech}")
    print(f"Caltech-101 Test accuracy (Top 5): {test_top_5_acc_caltech}")
    return model, history_caltech

# Re-encode Caltech-101 labels depending on loss function
def encode_caltech_labels(loss_function, dataset):
    if loss_function == 'categorical_crossentropy':
        return dataset.map(lambda image, label: (image, tf.one_hot(label, depth=101)))  # One-hot encoding for 101 classes
    return dataset  # Keep integer-encoded labels for sparse_categorical_crossentropy

train_dataset_encoded = encode_caltech_labels('categorical_crossentropy', train_dataset)
test_dataset_encoded = encode_caltech_labels('categorical_crossentropy', test_dataset)

try:
    cnn_model_caltech = create_cnn_model(input_shape=(128, 128, 3), num_classes=101, pooling_type='max')
    cnn_model_caltech_max, history_caltech_max = train_and_evaluate_caltech_model(cnn_model_caltech, train_dataset_encoded, test_dataset_encoded, epochs=ep)
except Exception as e:
    print(f"An error occurred: max pooling:create_cnn_model: {e}")

try:
    cnn_model_caltech_avg = create_cnn_model(input_shape=(128, 128, 3), num_classes=101, pooling_type='avg')
    cnn_model_caltech_avg, history_caltech_avg = train_and_evaluate_caltech_model(cnn_model_caltech_avg, train_dataset_encoded, test_dataset_encoded, epochs=ep)
except Exception as e:
    print(f"An error occurred: avg pooling:create_cnn_model: {e}")

### 6. Experiment with Different Loss Functions and Optimizers

def experiment_with_loss_and_optimizer(loss_function, optimizer_name):
    print(f"\nExperimenting with loss function: {loss_function} and optimizer: {optimizer_name}")

    # Re-encode MNIST labels
    y_train_encoded = encode_labels(loss_function, y_train)
    y_test_encoded = encode_labels(loss_function, y_test)

    # Train on MNIST
    cnn_model_mnist_exp = create_cnn_model(input_shape=(28, 28, 1), num_classes=10, pooling_type='max', loss_function=loss_function, optimizer_name=optimizer_name)
    history_mnist_exp = train_and_evaluate_model(cnn_model_mnist_exp, (x_train, y_train_encoded), (x_test, y_test_encoded), epochs=ep)

    # Train on Caltech-101
    train_dataset_encoded = encode_caltech_labels(loss_function, train_dataset)
    test_dataset_encoded = encode_caltech_labels(loss_function, test_dataset)

    cnn_model_caltech_exp = create_cnn_model(input_shape=(128, 128, 3), num_classes=101, pooling_type='max', loss_function=loss_function, optimizer_name=optimizer_name)
    history_caltech_exp = train_and_evaluate_caltech_model(cnn_model_caltech_exp, train_dataset_encoded, test_dataset_encoded, epochs=ep)

# Experiment with different configurations
experiment_with_loss_and_optimizer(loss_function='categorical_crossentropy', optimizer_name='adam')
experiment_with_loss_and_optimizer(loss_function='categorical_crossentropy', optimizer_name='sgd')
experiment_with_loss_and_optimizer(loss_function='sparse_categorical_crossentropy', optimizer_name='adam')
experiment_with_loss_and_optimizer(loss_function='sparse_categorical_crossentropy', optimizer_name='rmsprop')


### 8. Comparison of Pooling Methods

# Function to plot comparison of accuracies and losses
def plot_history(histories, title):
    plt.figure(figsize=(12, 6))
    
    # Plot training accuracy
    plt.subplot(1, 2, 1)
    for label, history in histories.items():
        plt.plot(history.history['accuracy'], label=label)
    plt.title(f"{title} - Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    for label, history in histories.items():
        plt.plot(history.history['val_accuracy'], label=label)
    plt.title(f"{title} - Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Compare results for MNIST
mnist_histories = {"Max Pooling": history_max_mnist, "Avg Pooling": history_avg_mnist}
plot_history(mnist_histories, "MNIST Pooling Comparison")

# Compare results for Caltech-101
caltech_histories = {"Max Pooling": history_caltech_max, "Avg Pooling": history_caltech_avg}
plot_history(caltech_histories, "Caltech-101 Pooling Comparison")



### 7. Feature Map Visualization (Task)

# Visualize the feature maps generated by the first and second convolutional layers
def visualize_feature_maps(model, image, layer_names):
    # Create a model that outputs feature maps from specified layers
    outputs = [model.get_layer(layer_name).output for layer_name in layer_names]
    feature_map_model = models.Model(inputs=model.input, outputs=outputs)

    # Generate feature maps
    feature_maps = feature_map_model.predict(image)

    # Plot feature maps for each layer
    for layer_name, feature_map in zip(layer_names, feature_maps):
        print(f"Visualizing feature maps for layer: {layer_name}")
        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]
        n_cols = n_features // 8
        display_grid = np.zeros((size, size * n_cols))

        for i in range(n_cols):
            feature = feature_map[0, :, :, i]
            feature -= feature.mean()
            feature /= feature.std() if feature.std() != 0 else 1
            feature *= 64
            feature += 128
            feature = np.clip(feature, 0, 255).astype('uint8')
            display_grid[:, i * size: (i + 1) * size] = feature

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(f"Feature maps for {layer_name}")
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

# Select one image from each dataset for feature map visualization
sample_image_mnist = np.expand_dims(x_train[0], axis=0)
sample_image_caltech = None
for images, _ in train_dataset.take(1):
    sample_image_caltech = np.expand_dims(images[0].numpy(), axis=0)

# Visualize feature maps after first and second convolutional layers for MNIST
layer_names = ["conv_1", "conv_2"]
print("Visualizing MNIST feature maps:")
visualize_feature_maps(cnn_model_mnist_max, sample_image_mnist, layer_names)

# Visualize feature maps after first and second convolutional layers for Caltech-101
print("Visualizing Caltech-101 feature maps:")
visualize_feature_maps(cnn_model_caltech_max, sample_image_caltech, layer_names)
