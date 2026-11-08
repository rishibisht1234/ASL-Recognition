
import numpy as np
import tensorflow as tf
import joblib

from config import CLASSIFIER_PATH, LABEL_ENCODER_PATH


class ASLPredictor:

    def __init__(self):

        self.model = tf.keras.models.load_model(
            CLASSIFIER_PATH
        )

        self.label_encoder = joblib.load(
            LABEL_ENCODER_PATH
        )

        print("ASL model loaded successfully!")

    def predict(self, features):
        """
        Predict ASL alphabet from normalized landmarks.
        """

        input_data = np.asarray(
            features, dtype=np.float32
        ).reshape(1, 63)

        probabilities = self.model.predict(
            input_data,
            verbose=0
        )[0]

        predicted_index = np.argmax(probabilities)

        predicted_letter = self.label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = float(
            probabilities[predicted_index] * 100
        )

        return predicted_letter, confidence
