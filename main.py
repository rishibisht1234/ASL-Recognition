
import cv2
import mediapipe as mp


import tensorflow as tf
import joblib

# Load trained model
model = tf.keras.models.load_model(
    "models/asl_mlp_baseline.keras"
)

# Load label encoder
label_encoder = joblib.load(
    "models/label_encoder.joblib"
)

print("Model loaded successfully!")


# -------------------------------
# 1. Initialize MediaPipe
# -------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)



import numpy as np

def normalize_landmarks(features):
    """
    Normalize 21 hand landmarks using wrist-relative
    coordinates and maximum wrist distance.
    """

    landmarks = np.asarray(features, dtype=np.float32).reshape(21, 3)

    # Wrist landmark (landmark 0)
    wrist = landmarks[0].copy()

    # Make all landmarks relative to wrist
    relative_landmarks = landmarks - wrist

    # Calculate distance of every landmark from wrist
    distances = np.linalg.norm(relative_landmarks, axis=1)

    # Find maximum distance for scale normalization
    scale = distances.max()

    if scale < 1e-6:
        raise ValueError("Invalid hand landmark scale")

    normalized = relative_landmarks / scale

    return normalized.flatten()


# -------------------------------
# 2. Start webcam
# -------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# -------------------------------
# 3. Create HandLandmarker
# -------------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to capture frame")
            break

        # Mirror frame for a natural webcam experience
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # VIDEO mode requires timestamps in milliseconds
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Detect hands
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        if len(result.hand_landmarks) == 1:

    # Extract the 21 landmarks
            hand_landmarks = result.hand_landmarks[0]

            # Convert landmarks into 63 features
            features = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float32
            ).flatten()

            # Normalize features
            normalized_features = normalize_landmarks(features)

            # Reshape features for model input
            input_data = normalized_features.reshape(1, 63)

            # Get prediction probabilities
            predictions = model.predict(input_data, verbose=0)

            # Get predicted class index
            predicted_index = np.argmax(predictions[0])

            # Convert index into alphabet
            predicted_letter = label_encoder.inverse_transform(
                [predicted_index]
            )[0]

            # Get prediction confidence
            confidence = np.max(predictions[0]) * 100

            print(
                f"Prediction: {predicted_letter} | "
                f"Confidence: {confidence:.2f}%"
            )

            cv2.putText(
                frame,
                f"Prediction: {predicted_letter}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}%",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )



            print("Original shape:", features.shape)
            print("Normalized shape:", normalized_features.shape)
            print("Normalized min:", normalized_features.min())
            print("Normalized max:", normalized_features.max())


        if len(result.hand_landmarks) == 0:
            cv2.putText(
                frame,
                "No hand detected",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        # -------------------------------
        # 4. Draw landmarks
        # -------------------------------

        h, w, _ = frame.shape

        for hand_landmarks in result.hand_landmarks:

            # Draw landmark points
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Display number of landmarks
            cv2.putText(
                frame,
                "21 landmarks detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Show hand count
        hand_count = len(result.hand_landmarks)

        cv2.putText(
            frame,
            f"Hands detected: {hand_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.imshow("ASL Hand Detection", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
