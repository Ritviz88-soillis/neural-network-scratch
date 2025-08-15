import numpy as np


def init_param(layer_dims):

    np.random.seed(42)  # For reproducibility

    w1 = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
    b1 = np.zeros((layer_dims[1], 1))
    w2 = np.random.randn(layer_dims[2], layer_dims[1]) * 0.01
    b2 = np.zeros((layer_dims[2], 1))

    return w1, b1, w2, b2


def ReLU(x):
    return np.maximum(0, x)


def softmax(x):

    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def forward_propagation(w1, b1, w2, b2, X):
    z1 = np.dot(w1, X) + b1
    a1 = ReLU(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2


def backward_propagation(X, Y, w1, b1, w2, b2, z1, a1, z2, a2):

    m = X.shape[1]
    # Output layer gradients
    dz2 = a2 - Y
    dw2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * (z1 > 0)  # ReLU derivative
    dw1 = np.dot(dz1, X.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate=0.01):

    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

    return w1, b1, w2, b2


def compute_loss(Y_true, Y_pred):
    """Compute cross-entropy loss"""
    m = Y_true.shape[1]
    loss = -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m
    return loss


def compute_accuracy(Y_true, Y_pred):
    """Compute accuracy"""
    predictions = np.argmax(Y_pred, axis=0)
    true_labels = np.argmax(Y_true, axis=0)
    accuracy = np.mean(predictions == true_labels)
    return accuracy


def gradient_descent(X_train, Y_train, X_test, Y_test, iterations, learning_rate=0.1):

    w1, b1, w2, b2 = init_param(layer_dims=[784, 128, 10])

    print("Starting training...")
    print("-" * 50)

    for i in range(iterations):
        # Forward propagation
        z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, X_train)

        # Compute loss and accuracy
        loss = compute_loss(Y_train, a2)
        train_acc = compute_accuracy(Y_train, a2)

        # Test accuracy
        _, _, _, test_pred = forward_propagation(w1, b1, w2, b2, X_test)
        test_acc = compute_accuracy(Y_test, test_pred)


        dw1, db1, dw2, db2 = backward_propagation(X_train, Y_train, w1, b1, w2, b2, z1, a1, z2, a2)

        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)

        # Print progress
        if i % 10 == 0:
            print(f"Iteration {i:3d} | Loss: {loss:.4f} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

    print("-" * 50)
    print("Training completed!")

    return w1, b1, w2, b2


def predict(w1, b1, w2, b2, X):
    """Make predictions on new data"""
    _, _, _, a2 = forward_propagation(w1, b1, w2, b2, X)
    predictions = np.argmax(a2, axis=0)
    return predictions