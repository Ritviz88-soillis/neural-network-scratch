import numpy as np
from data_loader import load_mnist_data, one_hot_encode
from neural_network import gradient_descent, predict, compute_accuracy


def save_values(w1, b1, w2, b2, weights="weights.txt", bais = "bais.txt"):
    with open(weights, 'w') as f:
        f.write("# Weight matrix W1\n")
        f.write(f"W1_shape: {w1.shape[0]},{w1.shape[1]}\n")
        for row in w1:
            f.write(' '.join(map(str, row)) + '\n')

        f.write("\n# Weight matrix W2\n")
        f.write(f"W2_shape: {w2.shape[0]},{w2.shape[1]}\n")
        for row in w2:
            f.write(' '.join(map(str, row)) + '\n')

    with open(bais, 'w') as f:
        f.write("# Bias vector B1\n")
        f.write(f"B1_shape: {b1.shape[0]},{b1.shape[1]}\n")
        for row in b1:
            f.write(' '.join(map(str, row)) + '\n')

        f.write("\n# Bias vector B2\n")
        f.write(f"B2_shape: {b2.shape[0]},{b2.shape[1]}\n")
        for row in b2:
            f.write(' '.join(map(str, row)) + '\n')

    print(f"Model saved successfully!")
    print(f"Weights saved to: {weights}")
    print(f"Biases saved to: {bais}")


def load_values( weights="weights.txt", bais = "bais.txt"):
    with open(weights, 'r') as f:
        lines = f.readlines()

    # Parse W1 and W2
    w1_data, w2_data = [], []
    w1_shape, w2_shape = None, None
    parsing_w1, parsing_w2 = False, False

    for line in lines:
        line = line.strip()
        if line.startswith("W1_shape:"):
            w1_shape = tuple(map(int, line.split(": ")[1].split(",")))
            parsing_w1, parsing_w2 = True, False
        elif line.startswith("W2_shape:"):
            w2_shape = tuple(map(int, line.split(": ")[1].split(",")))
            parsing_w1, parsing_w2 = False, True
        elif line and not line.startswith("#") and not line.startswith("W"):
            if parsing_w1:
                w1_data.append([float(x) for x in line.split()])
            elif parsing_w2:
                w2_data.append([float(x) for x in line.split()])

    w1 = np.array(w1_data).reshape(w1_shape)
    w2 = np.array(w2_data).reshape(w2_shape)

    # Load biases
    with open(bais, 'r') as f:
        lines = f.readlines()

    # Parse B1 and B2
    b1_data, b2_data = [], []
    b1_shape, b2_shape = None, None
    parsing_b1, parsing_b2 = False, False

    for line in lines:
        line = line.strip()
        if line.startswith("B1_shape:"):
            b1_shape = tuple(map(int, line.split(": ")[1].split(",")))
            parsing_b1, parsing_b2 = True, False
        elif line.startswith("B2_shape:"):
            b2_shape = tuple(map(int, line.split(": ")[1].split(",")))
            parsing_b1, parsing_b2 = False, True
        elif line and not line.startswith("#") and not line.startswith("B"):
            if parsing_b1:
                b1_data.append([float(x) for x in line.split()])
            elif parsing_b2:
                b2_data.append([float(x) for x in line.split()])

    b1 = np.array(b1_data).reshape(b1_shape)
    b2 = np.array(b2_data).reshape(b2_shape)

    print(f"Model loaded successfully!")
    print(f"W1 shape: {w1.shape}, W2 shape: {w2.shape}")
    print(f"B1 shape: {b1.shape}, B2 shape: {b2.shape}")

    return w1, b1, w2, b2


def main():
    print("MNIST Digit Recognition Neural Network")


    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data()

    print(f"\nData loaded successfully!")
    print(f"Training set: {train_images.shape[0]} images")
    print(f"Test set: {test_images.shape[0]} images")
    print(f"Image size: {train_images.shape[1]} pixels")


    X_train = train_images.T  # Shape: (784, 60000)
    X_test = test_images.T  # Shape: (784, 10000)


    Y_train = one_hot_encode(train_labels)  # Shape: (10, 60000)
    Y_test = one_hot_encode(test_labels)  # Shape: (10, 10000)

    print(f"\nData shapes after preprocessing:")
    print(f"X_train: {X_train.shape}")
    print(f"Y_train: {Y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"Y_test: {Y_test.shape}")

    print(f"\nSample training images and labels:")
    print(f"\nSample labels: {train_labels[:10]}")


    # Training parameters
    iterations = 1000
    learning_rate = 0.1

    print(f"\nTraining Parameters:")
    print(f"Iterations: {iterations}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Network Architecture: 784 → 128 → 10")

    # Train the model
    w1, b1, w2, b2 = gradient_descent(
        X_train, Y_train, X_test, Y_test,
        iterations=iterations,
        learning_rate=learning_rate
    )
    print(f"\nSaving trained model...")
    save_values(w1, b1, w2, b2, "weights.txt", "bais.txt")
    # Final evaluation
    print(f"\nFinal Evaluation:")

    # Training accuracy
    train_predictions = predict(w1, b1, w2, b2, X_test)
    train_accuracy = np.mean(train_predictions == test_labels)
    print(f"Training Accuracy: {train_accuracy:.3f}")

    # Test accuracy
    test_predictions = predict(w1, b1, w2, b2, X_test)
    test_accuracy = np.mean(test_predictions == test_labels)
    print(f"Test Accuracy: {test_accuracy:.3f}")

    # Show some predictions
    print(f"\nSample Predictions vs True Labels:")
    print("Predicted:", test_predictions[:10])
    print("True:     ", test_labels[:10])
    print("Match:    ", test_predictions[:10] == test_labels[:10])

    # Show prediction distribution
    print(f"\nPrediction Distribution:")
    pred_counts = np.bincount(test_predictions, minlength=10)
    true_counts = np.bincount(test_labels, minlength=10)

    print("Digit | Predicted | True")
    print("-" * 25)
    for digit in range(10):
        print(f"  {digit}   |    {pred_counts[digit]:4d}   | {true_counts[digit]:4d}")

    return w1, b1, w2, b2


if __name__ == "__main__":
    # Run the main training pipeline
    trained_model = main()