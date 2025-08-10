import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
import numpy as np


def build_nn_model(hp, input_dim):
    """Build and compile a keras model with hyperparameters"""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(hp.Int("units1", 64, 512, step=64), activation="relu"),
        keras.layers.Dropout(hp.Float("dropout1", 0.0, 0.5, step=0.1)),
        keras.layers.Dense(hp.Int("units2", 64, 512, step=64), activation="relu"),
        keras.layers.Dropout(hp.Float("dropout2", 0.0, 0.5, step=0.1)),
        keras.layers.Dense(1, activation="linear"),  # regression
    ])

    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",  # <-- critical: define a loss for regression
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model

def train_neural_network(X, y, param_name):
    """Train the neural network model."""
    input_dim = X.shape[1]
    # Define Keras Tuner with RandomSearch
    tuner = kt.RandomSearch(
        lambda hp: build_nn_model(hp, input_dim),
        objective=kt.Objective('val_rmse', direction='min'),
        max_trials=15,
        overwrite=True,
        directory='data/kt_tuner_dir',
        project_name=f'{param_name}_tuning'
    )

    # Define callbacks: EarlyStopping and learning rate scheduler
    early_stop = keras.callbacks.EarlyStopping(monitor='val_rmse', mode='min', patience=5, restore_best_weights=True)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_rmse', mode='min', factor=0.5, patience=5)

    # Run Hyperparameter search
    tuner.search(
        X, y,
        epochs=50,
        validation_split=0.2,
        callbacks=[early_stop, lr_scheduler],
        verbose=2,
        batch_size=512,
    )

    # Retrieve the best model
    best_model = tuner.get_best_models(1)[0]

    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(1)[0]

    # Evaluate on the validation set
    hist = best_model.fit(
        X, y, validation_split=0.2, epochs=1, batch_size=512, verbose=0
    )
    val_rmse = float(hist.history.get("val_rmse", [np.nan])[-1])

    # Return the best model and its hyperparameters
    return best_model, best_hps.values, val_rmse

def predict_with_nn(model, X):
    """Predict using the trained neural network model."""
    return model.predict(X)

