import tensorflow as tf
import keras_tuner as kt
from tf.keras.models import Sequential
from tf.keras.layers import Dense, Dropout
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from tf.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_nn_model(hp, input_shape, use_physics_loss=False):
    """Build and compile a keras model with hyperparameters"""
    model = Sequential()
    model.add(tf.keras.Input(shape=(input_shape,)))
    # First hidden layer: tune number of units and dropout rate.
    units1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    model.add(Dense(units1, activation='relu'))
    if dropout_rate > 0.0:
        model.add(Dropout(dropout_rate))

    # Second hidden layer: tune number of units.
    units2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
    model.add(Dense(units2, activation='relu'))
    if dropout_rate > 0.0:
        model.add(Dropout(dropout_rate))

    # Output layer (single neuron for regression)
    model.add(Dense(1))

    # Tune the learning rate for the optimizer
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compile the model with the loss function
    model.compile(optimizer=optimizer, loss=loss_function(hp, use_physics_loss=use_physics_loss))
    
    return model

def train_neural_network(X, y, param_name):
    """Train the neural network model."""
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define Keras Tuner with RandomSearch
    tuner = kt.RandomSearch(
        lambda hp: build_nn_model(hp, input_shape=X_train.shape[1], use_physics_loss=True),
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory='kt_tuner_dir',
        project_name=f'{param_name}_tuning'
    )

    # Define callbacks: EarlyStopping and learning rate scheduler
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    # Run Hyperparameter search
    tuner.search(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, lr_scheduler],
        verbose=2
    )

    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Evaluate on the validation set
    val_predictions = best_model.predict(X_val)
    val_rmse = root_mean_squared_error(y_val, val_predictions)

    # Return the best model and its hyperparameters
    return best_model, best_hps.values, val_rmse

def predict_with_nn(model, X):
    """Predict using the trained neural network model."""
    return model.predict(X)

def compute_physics_loss(y_pred, min_val=-2.0, max_val=2.0):
    """Calculates the physics loss for predictions outside the specified range."""
    lower_violation = tf.nn.relu(min_val - y_pred)  # Penalize predictions below min_val
    upper_violation = tf.nn.relu(y_pred - max_val)  # Penalize predictions above max_val
    physics_loss = tf.reduce_mean(tf.square(lower_violation) + tf.square(upper_violation))
    return physics_loss

def loss_function(hp=None, min_val=-2.0, max_val=2.0, use_physics_loss=False):
    """
    Creates a loss function that optionally combines data loss and physics loss.
    
    Args:
        hp: Hyperparameter object (optional, for tuning physics loss weight).
        min_val: Minimum valid value for predictions.
        max_val: Maximum valid value for predictions.
        use_physics_loss: Whether to include physics loss in the combined loss.

    Returns:
        A loss function to be used in model compilation.
    """
    # Set the physics loss weight (tunable if hp is provided)
    physics_loss_weight = hp.Float('physics_loss_weight', min_value=0.01, max_value=1.0, sampling='LOG') if hp else 1.0

    def loss(y_true, y_pred):
        """Combined loss function."""
        # Data loss (mean squared error)
        data_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        if use_physics_loss:
            # Physics loss
            physics_loss = compute_physics_loss(y_pred, min_val, max_val)
            # Combine data loss and physics loss
            return data_loss + physics_loss_weight * physics_loss
        else:
            # Return only data loss if physics loss is not used
            return data_loss

    return loss
