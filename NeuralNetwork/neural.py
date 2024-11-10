# Import required libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the MNIST dataset from the local file
mnist_path = '/Users/udeesharukshan/Documents/GenAI/NeuralNetwork/mnist.npz'
with np.load(mnist_path) as data:
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

# Normalize the images to [0, 1] range
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 2: Build the Neural Network Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images into 1D vector
    tf.keras.layers.Dense(128, activation='relu'),  # First dense layer with 128 neurons
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 classes (0-9)
])

# Step 3: Compile the Model
model.compile(optimizer='adam',  # Optimizer: Adam
              loss='sparse_categorical_crossentropy',  # Loss function
              metrics=['accuracy'])  # Metric: accuracy

# Step 4: Train the Model
model.fit(x_train, y_train, epochs=5)  # Train for 5 epochs

# Step 5: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test)  # Evaluate on test set
print('\nTest accuracy:', test_acc)

# Step 6: Make Predictions
predictions = model.predict(x_test)

# Function to plot a single image with predicted and true labels
def plot_image(predictions_array, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"{predicted_label} ({true_label})", color=color)

# Plot the first 5 test images with predictions
num_rows, num_cols = 1, 5
plt.figure(figsize=(2 * num_cols, 2 * num_rows))
for i in range(5):
    plt.subplot(num_rows, num_cols, i + 1)
    plot_image(predictions[i], y_test[i], x_test[i])
plt.show()
