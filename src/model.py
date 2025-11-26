import tensorflow as tf
import numpy as np

def build_uplift_model(input_dim=3, hidden_units=32, dropout_rate=0.1):
    """
    Build a simple regression model for predictin uplift_score.
    input_dim = numbe rof input features
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(hidden_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(hidden_units, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear") # regression output
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model

def train_uplift_model(model, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
    """
    Train the uplift model.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

    return history, model


def save_uplift_model(model, path="models/uplift_model.keras"):
    """
    Save TensorFlow model to disk.
    """
    model.save(path)
    print(f"Model saved to {path}")
