import numpy as np
import tensorflow as tf
from src.features import compute_crime_rate, compute_uplift_score

MODEL_PATH = "models/uplift_model.keras"

def load_tf_model(path: str = MODEL_PATH):
    """
    Load the TensorFlow uplift model.
    """
    model = tf.keras.models.load_model(path)
    return model

def prepare_input_features(crime_count: float,
                           population: float,
                           accessibility_score: float,
                           home_value_score: float):
        """
        Convert raw inputs into model-ready features.
        """
        # 1. Compute crime rate per 1000 people
        crime_rate = compute_crime_rate(crime_count, population)

        # 2. Features must be passed to model in correct order
        features = np.array([[crime_rate,
                              accessibility_score,
                              home_value_score]])
        
        return features

def predict_uplift(model,
                   crime_count: float,
                   population: float,
                   accessibility_score: float,
                   home_value_score: float):
      """
      Full prediction pipeline.
      """

      X = prepare_input_features(
            crime_count,
            population,
            accessibility_score,
            home_value_score
      )

      uplift_pred = model.predict(X)[0][0] # model outputs shape: (1, 1)

      return float(uplift_pred)

# Optional local test
if __name__ == "__main__":
      model = load_tf_model()

      test_pred = predict_uplift(
            model,
            crime_count=10,
            population=5000,
            accessibility_score=0.5,
            home_value_score=0.7
      )

      print("Test uplift prediction:", test_pred)