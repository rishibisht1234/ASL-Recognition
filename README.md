# ASL Recognition — Real-Time Sign Language Alphabet Recognition

A real-time American Sign Language (ASL) alphabet recognition system built using **MediaPipe, TensorFlow/Keras, and OpenCV**.

The system detects hand landmarks from a webcam, preprocesses them into normalized features, and uses a trained Multi-Layer Perceptron (MLP) to classify 26 ASL alphabet classes.

## Model Performance

| Metric | Result |
|---|---|
| Test Accuracy | 98.78% |
| Test Loss | 0.0447 |
| Macro F1-Score | 98.67% |
| Number of Classes | 26 |
| Input Features | 63 |

*Metrics are measured on a stratified random test split. Performance on unseen signers and environments may differ.*

## Features

- Real-time webcam-based hand detection
- MediaPipe Tasks API for 21 hand landmarks
- Wrist-relative and scale-normalized landmark features
- MLP-based alphabet classification
- Confidence-weighted temporal prediction smoothing
- Live prediction and confidence display
- Modular Python architecture

## Technology Stack

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn
- Joblib
- uv

## How It Works

1. Capture video frames using OpenCV.
2. Detect hand landmarks using MediaPipe.
3. Extract 21 × 3 = 63 landmark features.
4. Normalize landmarks relative to the wrist and hand scale.
5. Predict the ASL alphabet using the trained MLP.
6. Apply temporal smoothing to stabilize predictions.
7. Display the prediction on the webcam feed.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ASL-Recognition.git
cd ASL-Recognition
```

Install dependencies:

```bash
uv sync
```


Ensure the trained classifier and label encoder are present in the `models/` directory.

## Usage

```bash
uv run python main.py
```

Show one hand to the webcam to begin recognition. Press **Q** to exit.

## Limitations

- Currently supports 26 alphabet classes only.
- J and Z require motion for complete ASL recognition, which is not captured by the single-frame classifier.
- Similar hand configurations, particularly M and N, can be confused.
- Real-world performance may vary with hand orientation, camera conditions, and signer differences.

## Future Improvements

- Improved recognition of visually similar signs
- Dynamic gesture recognition
- Letter-to-word formation
- Sentence generation
- Improved real-time prediction stability
