import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

def build_nn_model(input_shape):
    """Build a simple feedforward neural network model."""
    model = Sequential()

    # First hidden layer
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dropout(0.2))  # Regularization layer to prevent overfitting

    # Second hidden layer
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    # Output layer (single neuron for regression)
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_neural_network(X, y):
    """Train the neural network model."""
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build the model
    model = build_nn_model(X_train.shape[1])

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=2)

    # Evaluate the model
    val_predictions = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Validation RMSE: {val_rmse}")

    return model, history

def predict_with_nn(model, X):
    """Predict using the trained neural network model."""
    return model.predict(X)
