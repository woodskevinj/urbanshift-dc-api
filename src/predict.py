import numpy as np
import tensorflow as tf
# from src.features import compute_crime_rate, compute_uplift_score

MODEL_PATH = "models/uplift_model.keras"

def load_tf_model(path: str = MODEL_PATH):
    """
    Load the TensorFlow uplift model.
    """
    model = tf.keras.models.load_model(path)
    return model

def prepare_input_features(
            crime_count: float,
            population: float,
            accessibility_score: float,
            home_value_score: float):
        """
        Convert raw inputs into model-ready features.
        - crime_rate_per_1000 = (crime_count / population) * 1000
        Returns a (1, 3) numpy array for model.predict().
        """
        if population <= 0:
              raise ValueError("Poulation must be positive.")
        
        crime_rate_per_1000 = (crime_count / population) * 1000

        features = np.array(
              [[crime_rate_per_1000, accessibility_score, home_value_score]],
              dtype=np.float32,
        )
        
        return features, crime_rate_per_1000

def predict_uplift(
            model,
            crime_count: float,
            population: float,
            accessibility_score: float,
            home_value_score: float,
):
      """
      Full prediction pipeline: raw inputs -> features -> uplift score.
      """

      X, crime_rate_per_1000 = prepare_input_features(
            crime_count,
            population,
            accessibility_score,
            home_value_score
      )

      preds = model.predict(X, verbose=0)

      uplift_pred = float(preds[0][0])

      return uplift_pred, crime_rate_per_1000

# Optional local test
if __name__ == "__main__":
      m = load_tf_model()
      uplift, cr = predict_uplift(
            m,
            crime_count=10,
            population=5000,
            accessibility_score=0.5,
            home_value_score=0.7
      )

      print("TEst crime_rate_per_1000:", cr)
      print("Test uplift prediction:", uplift)