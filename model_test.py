import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_mnist_data
from neural_network import predict, forward_propagation
from main import main

def create_test_digit(digit, noise_level=0.1):
    image = np.zeros((28, 28))

    if digit == 0:
        # Create a circle for 0
        center = (14, 14)
        for i in range(28):
            for j in range(28):
                dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                if 8 < dist < 12:
                    image[i, j] = 1.0

    elif digit == 1:
        # Create a vertical line for 1
        image[4:24, 13:15] = 1.0
        image[4:8, 11:13] = 1.0  # Top diagonal

    elif digit == 7:
        # Create a 7
        image[4:6, 6:22] = 1.0  # Top horizontal
        image[6:24, 16:18] = 1.0  # Diagonal down

    else:
        # For other digits, create a simple pattern
        np.random.seed(digit * 42)
        image = np.random.rand(28, 28) * 0.8
        image = (image > 0.5).astype(float)
    # Add noise
    noise = np.random.rand(28, 28) * noise_level
    image = np.clip(image + noise, 0, 1)

    return image


def test_with_synthetic_data(w1, b1, w2, b2):

    print("\n" + "=" * 60)
    print("TESTING WITH SYNTHETIC DIGIT PATTERNS")
    print("=" * 60)

    test_digits = [0, 1, 7]  # Digits we created patterns for

    for digit in test_digits:
        # Create test image
        test_image = create_test_digit(digit)

        # Flatten and transpose for neural network
        test_input = test_image.flatten().reshape(784, 1)

        # Make prediction
        prediction = predict(w1, b1, w2, b2, test_input)

        # Get confidence scores
        _, _, _, probs = forward_propagation(w1, b1, w2, b2, test_input)
        confidence = np.max(probs) * 100

        print(f"Created digit: {digit}")
        print(f"Model prediction: {prediction[0]}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"Match: {'✓' if prediction[0] == digit else '✗'}")

        # Show top 3 predictions
        top_3_idx = np.argsort(probs.flatten())[-3:][::-1]
        top_3_probs = probs.flatten()[top_3_idx] * 100
        print(
            f"Top 3: {top_3_idx[0]}({top_3_probs[0]:.1f}%), {top_3_idx[1]}({top_3_probs[1]:.1f}%), {top_3_idx[2]}({top_3_probs[2]:.1f}%)")
        print("-" * 40)


def test_with_random_data(w1, b1, w2, b2):

    print("\n" + "=" * 60)
    print("TESTING WITH RANDOM NOISE")
    print("=" * 60)
    num_tests = 5

    for i in range(num_tests):
        # Create random image
        np.random.seed(i)
        random_image = np.random.rand(28, 28)
        # Flatten and transpose
        test_input = random_image.flatten().reshape(784, 1)

        prediction = predict(w1, b1, w2, b2, test_input)

        _, _, _, probs = forward_propagation(w1, b1, w2, b2, test_input)
        confidence = np.max(probs) * 100

        print(f"Random test {i + 1}:")
        print(f"Prediction: {prediction[0]} (confidence: {confidence:.1f}%)")

        if confidence > 50:
            print("Model is confident (might be overfitting)")
        else:
            print("Model is uncertain (good for random noise)")
        print("-" * 30)


def visualize_predictions(w1, b1, w2, b2):
    """Visualize some predictions with images"""
    print("\n" + "=" * 60)
    print("VISUAL PREDICTION TEST")
    print("=" * 60)

    (_, _), (test_images, test_labels) = load_mnist_data()
    indices = [0, 1, 2, 100, 500]
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 3))

    for i, idx in enumerate(indices):
        image = test_images[idx].reshape(28, 28)
        test_input = test_images[idx].reshape(784, 1)

        prediction = predict(w1, b1, w2, b2, test_input)
        true_label = test_labels[idx]

        _, _, _, probs = forward_propagation(w1, b1, w2, b2, test_input)
        confidence = np.max(probs) * 100
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'True: {true_label}\nPred: {prediction[0]}\n({confidence:.0f}%)')
        axes[i].axis('off')

        if prediction[0] == true_label:
            axes[i].title.set_color('green')
        else:
            axes[i].title.set_color('red')

    plt.tight_layout()
    plt.savefig('prediction_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Prediction visualization saved as 'prediction_test.png'")


def main_test():
    print("MNIST MODEL TESTING SUITE")
    print("=" * 60)

    print("Training model first...")
    w1, b1, w2, b2 = main()  # This will train and return the model

    test_with_synthetic_data(w1, b1, w2, b2)
    test_with_random_data(w1, b1, w2, b2)
    try:
        visualize_predictions(w1, b1, w2, b2)
    except ImportError:
        print("\nMatplotlib not available - skipping visualization")
    except Exception as e:
        print(f"\nVisualization failed: {e}")

    print("\n" + "=" * 60)
    print("TESTING COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main_test()