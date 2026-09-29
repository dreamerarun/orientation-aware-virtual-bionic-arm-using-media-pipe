# 🦾 Orientation-Aware Virtual Bionic Arm Using MediaPipe

A real-time **hand gesture recognition and virtual bionic hand system** that uses **MediaPipe Hands**, machine learning, and **3D hand orientation features (Yaw, Pitch, Roll)** to recognize gestures and visualize them through a virtual bionic hand.

🔗 **Repository:** [orientation-aware-virtual-bionic-arm-using-media-pipe](https://github.com/dreamerarun/orientation-aware-virtual-bionic-arm-using-media-pipe)

The key idea of this project is to go beyond conventional hand-pose recognition by treating **hand orientation as part of the model's core feature vector** rather than simply logging it as additional information.

---

## ✨ Key Features

* 🖐️ Real-time hand landmark detection using **MediaPipe**
* 🧠 Random Forest-based gesture classification
* 📐 Integrated **Yaw, Pitch, and Roll** orientation features
* 📊 Automatic dataset generation and CSV storage
* 👤 Multi-user and multi-session dataset management
* 🔄 Landmark smoothing for more stable predictions
* 🎯 Real-time gesture confidence estimation
* 📈 Training/test accuracy and classification reports
* 🔢 Confusion matrix generation
* 🔍 Feature-importance analysis
* 💾 Automatic model, scaler, and metadata saving
* 🦾 Virtual bionic hand visualization using **Pygame**
* 📐 Live orientation bars for yaw, pitch, and roll
* 📝 Real-time prediction logging
* 📊 Dataset statistics and orientation analysis

---

## 🎯 Supported Gestures

The system currently recognizes five gestures:

| Label | Gesture     | Key |
| ----: | ----------- | --- |
|   `0` | ✋ Rest      | `R` |
|   `1` | ✊ Grasp     | `G` |
|   `2` | 🖐️ Release | `L` |
|   `3` | ☝️ Point    | `P` |
|   `4` | 🤏 Pinch    | `N` |

The gesture mapping is defined in the Python program:

```python
GESTURE_NAMES = {
    0: "rest",
    1: "grasp",
    2: "release",
    3: "point",
    4: "pinch"
}
```

---

# 🧠 System Overview

The overall pipeline is:

```text
                Webcam
                   │
                   ▼
          ┌─────────────────┐
          │ MediaPipe Hands │
          └────────┬────────┘
                   │
                   ▼
          21 Hand Landmarks
                   │
                   ▼
       ┌───────────────────────┐
       │ Feature Extraction    │
       ├───────────────────────┤
       │ Normalized Coordinates│
       │ Distances             │
       │ Joint Angles          │
       │ Palm Center           │
       │ Palm Spread           │
       │ Yaw                   │
       │ Pitch                 │
       │ Roll                  │
       └───────────┬───────────┘
                   │
                   ▼
           StandardScaler
                   │
                   ▼
          Random Forest Model
                   │
                   ▼
          Gesture Prediction
                   │
          ┌────────┴────────┐
          ▼                 ▼
   Camera Visualization   Virtual
                         Bionic Hand
```

---

# 📐 Orientation Integration

The major feature of this project is the integration of hand orientation into the machine-learning feature vector.

Three orientation parameters are calculated:

### Yaw

Represents the left-right rotation of the hand.

```text
Yaw → Left ↔ Right rotation
```

### Pitch

Represents the upward/downward tilt.

```text
Pitch → Up ↕ Down
```

### Roll

Represents the twisting rotation of the hand.

```text
Roll → Clockwise ↔ Counter-clockwise twist
```

These values are calculated directly from MediaPipe landmarks.

```python
orient = landmarks_get_orientation(lm)

orientation_features = [
    orient['yaw'],
    orient['pitch'],
    orient['roll']
]

features.extend(orientation_features)
```

Therefore, orientation is **not treated merely as metadata**.

It becomes part of the actual input used by the Random Forest classifier.

---

# 🧮 Feature Engineering

The model combines several types of hand features.

## 1. Normalized Landmark Coordinates

The 21 MediaPipe landmarks are represented relative to the wrist.

```python
norm = (lm - wrist).flatten()
```

This produces:

```text
21 landmarks × 3 coordinates = 63 features
```

---

## 2. Inter-Landmark Distances

Distances between important landmark pairs are calculated to capture hand geometry.

Examples include:

```text
Thumb ↔ Index
Thumb ↔ Middle
Index ↔ Middle
Wrist ↔ Thumb
Wrist ↔ Index
Wrist ↔ Middle
```

---

## 3. Finger Joint Angles

Finger joint angles are calculated using vectors between consecutive landmarks.

These angles help distinguish different finger configurations such as:

* Open hand
* Closed hand
* Pointing
* Pinching
* Grasping

---

## 4. Palm Center

The palm center is estimated using:

```python
palm_idx = [0, 5, 9, 13, 17]
```

The average position provides additional spatial information.

---

## 5. Orientation

Three additional features are included:

```text
Yaw
Pitch
Roll
```

These are integrated directly into the model input.

---

## 6. Palm Spread

The maximum distance between fingertips is calculated as an additional geometric feature.

---

# 🤖 Machine Learning Model

The project uses a **Random Forest Classifier** from Scikit-learn.

Current configuration:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=18,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

Before training, the complete feature vector is normalized using:

```python
StandardScaler()
```

The dataset is then divided into training and testing sets.

```text
80% → Training
20% → Testing
```

A stratified split is used when the class distribution allows it.

---

# 📊 Model Evaluation

The training pipeline provides several evaluation metrics:

### Training Accuracy

Measures performance on the training dataset.

### Test Accuracy

Measures performance on previously unseen test samples.

### Cross-Validation

When sufficient data is available, the system performs cross-validation.

### Classification Report

Includes:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

Shows which gestures are correctly classified and which gestures are confused with one another.

---

# 🔍 Feature Importance

Random Forest feature importance is also analyzed.

The program identifies the most influential features:

```python
feature_importances = self.model.feature_importances_
```

The top 15 features are displayed during training.

This makes it possible to investigate whether:

* Landmark positions
* Distances
* Joint angles
* Orientation
* Palm spread

are contributing strongly to gesture recognition.

---

# 📂 Dataset Structure

The project automatically generates several CSV files.

### Master Dataset

```text
gesture_master_dataset_with_orientation.csv
```

Contains all historical sessions.

### Session Dataset

```text
session_orientation_<SESSION_ID>.csv
```

Each recording session gets its own CSV file.

For example:

```text
session_orientation_20260929_183000.csv
```

### Prediction Log

```text
gesture_predictions_with_orientation.csv
```

Stores real-time predictions and orientation values.

Example structure:

```text
timestamp
pred_label
pred_conf
yaw
pitch
roll
```

---

# 💾 Model Files

After training, three files are generated.

```text
gesture_rf_model_orientation.pkl
gesture_scaler_orientation.pkl
training_metadata_orientation.pkl
```

### `gesture_rf_model_orientation.pkl`

Contains the trained Random Forest model.

### `gesture_scaler_orientation.pkl`

Contains the fitted `StandardScaler`.

### `training_metadata_orientation.pkl`

Stores information such as:

* Training sample count
* Test sample count
* Training accuracy
* Test accuracy
* Training timestamp
* Feature information
* Orientation integration status

---

# 📦 Requirements

## Hardware

Minimum recommended setup:

* 💻 Computer/Laptop
* 📷 Webcam
* 🖐️ One hand for gesture input

A dedicated GPU is **not required** for this project.

---

## Software

Recommended:

* Python 3.9+
* OpenCV
* MediaPipe
* NumPy
* Pandas
* Scikit-learn
* Pygame
* SciPy

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/dreamerarun/orientation-aware-virtual-bionic-arm-using-media-pipe.git
cd orientation-aware-virtual-bionic-arm-using-media-pipe
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it:

### Linux/macOS

```bash
source venv/bin/activate
```

### Windows

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn pygame scipy
```

---

# ▶️ Running the Project

Run the Python program:

```bash
python3 enhanced_bionic_gestures_with_orientation.py
```

You will see a menu similar to:

```text
============================================================
        ORIENTATION-AWARE BIONIC GESTURE SYSTEM
============================================================

MENU OPTIONS:

1. Collect new gesture data
2. Train model using CURRENT session only
3. Train model using ALL historical data
4. Load existing model from disk
5. Run real-time recognition & visualization
6. View dataset statistics
7. Test orientation impact
8. Exit
```

---

# 📝 Step 1 — Collect Gesture Data

Select:

```text
1
```

Enter your name when prompted.

Then choose the number of samples per gesture.

For example:

```text
Samples per gesture [120]: 200
```

The webcam window will then allow you to record:

```text
R → Rest
G → Grasp
L → Release
P → Point
N → Pinch
```

During recording, the system displays:

```text
Yaw
Pitch
Roll
```

Try to record each gesture from multiple orientations.

For example:

```text
        Hand Orientation

          Pitch
            ↑
            │
     Roll ← ✋ → Roll
            │
            ↓
          Pitch

       Yaw: rotate left/right
```

This helps the dataset contain variation rather than learning each gesture from only one fixed pose.

---

# 🏋️ Step 2 — Train the Model

After collecting data, choose:

```text
3
```

to train using all historical data.

Alternatively:

```text
2
```

trains using only the current session.

The training process:

```text
CSV Dataset
     ↓
Feature Extraction
     ↓
Orientation Included
     ↓
NaN Filtering
     ↓
StandardScaler
     ↓
Train/Test Split
     ↓
Random Forest
     ↓
Evaluation
     ↓
Save Model
```

---

# 🦾 Step 3 — Run Real-Time Recognition

After training, choose:

```text
5
```

The system opens:

1. Webcam visualization
2. Virtual bionic hand visualization

The webcam displays:

```text
Gesture: grasp (96.3%)

Yaw:   32.4°
Pitch: -11.7°
Roll:   18.5°
```

The Pygame interface additionally shows:

* Current gesture
* Prediction confidence
* Yaw bar
* Pitch bar
* Roll bar
* Virtual hand
* Gesture legend
* Model information
* Orientation status

---

# 🔄 Landmark Smoothing

To reduce jitter in the hand tracking, recent landmark positions are stored using a deque.

```python
SMOOTHING_QUEUE = 6
```

The current landmarks are averaged across the recent history:

```python
avg_land = np.mean(
    np.stack(list(self.landmark_history)),
    axis=0
)
```

Orientation values are also smoothed:

```python
ORIENTATION_SMOOTH = 0.6
```

This makes real-time visualization and predictions more stable.

---

# 📊 Dataset Statistics

The system can display:

```text
Total samples
Unique users
Total sessions
Gesture distribution
Orientation mean
Orientation standard deviation
Orientation range
```

Example:

```text
Gesture distribution:

rest       : 240 samples
grasp      : 240 samples
release    : 240 samples
point      : 240 samples
pinch      : 240 samples
```

Orientation statistics include:

```text
YAW    : Mean / Std / Min / Max
PITCH  : Mean / Std / Min / Max
ROLL   : Mean / Std / Min / Max
```

---

# 🧪 Orientation Impact Experiment

The menu includes an experimental option for evaluating orientation impact.

The intended experiment is:

```text
Model A
Landmarks + Geometry
       ↓
Accuracy

          VS

Model B
Landmarks + Geometry + Orientation
       ↓
Accuracy
```

This can help determine whether adding yaw, pitch, and roll provides measurable improvements for the selected dataset.

> Note: The current implementation always includes orientation. The menu option describes the experiment but does not automatically train the orientation-free comparison model.

---

# 🗂️ Project Structure

A typical project directory may look like:

```text
orientation-aware-virtual-bionic-arm-using-media-pipe/
│
├── enhanced_bionic_gestures_with_orientation.py
│
├── gesture_master_dataset_with_orientation.csv
├── gesture_predictions_with_orientation.csv
│
├── session_orientation_*.csv
│
├── gesture_rf_model_orientation.pkl
├── gesture_scaler_orientation.pkl
├── training_metadata_orientation.pkl
│
└── README.md
```

Generated datasets and model files can become large over time, so they should generally be excluded from Git if they are not intended to be version controlled.

Recommended `.gitignore` entries:

```gitignore
venv/
__pycache__/
*.pyc

*.csv
*.pkl

session_orientation_*.csv
gesture_master_dataset_with_orientation.csv
gesture_predictions_with_orientation.csv
```

---

# 🔬 Technical Details

### Hand Detection

**MediaPipe Hands**

Detects:

```text
21 landmarks
```

for the detected hand.

### Feature Processing

The system combines:

```text
63 normalized landmark coordinates
+ distance features
+ finger joint angles
+ palm center
+ yaw
+ pitch
+ roll
+ palm spread
```

### Classifier

```text
Random Forest
200 estimators
Maximum depth = 18
```

### Real-Time Processing

The pipeline operates continuously:

```text
Camera
 ↓
MediaPipe
 ↓
Landmarks
 ↓
Smoothing
 ↓
Feature Extraction
 ↓
Orientation Calculation
 ↓
Scaling
 ↓
Random Forest
 ↓
Gesture + Confidence
 ↓
Virtual Bionic Hand
```

---

# 🦾 Potential Applications

This project can serve as a foundation for:

* Virtual prosthetic hand interfaces
* Bionic hand control
* Human-robot interaction
* Gesture-controlled robotics
* Assistive technology
* Robot manipulation interfaces
* Human-machine interfaces
* Physical AI experiments
* Vision-based robotic control
* Hand gesture teleoperation

A future version could map recognized gestures directly to a simulated or physical robotic/bionic hand.

For example:

```text
Rest    → Neutral position
Grasp   → Close fingers
Release → Open fingers
Point   → Index extension
Pinch   → Thumb-index motion
```

---

# 🚀 Future Improvements

Possible extensions include:

* [ ] Add more hand gestures
* [ ] Support two-hand recognition
* [ ] Add temporal gesture recognition
* [ ] Use LSTM/GRU/Transformer models
* [ ] Compare Random Forest with neural networks
* [ ] Implement automatic orientation-impact benchmarking
* [ ] Add gesture recording quality checks
* [ ] Improve orientation estimation using 3D rotation matrices
* [ ] Add robotic hand/servo control
* [ ] Connect to ROS 2
* [ ] Control a simulated robot hand in Isaac Sim
* [ ] Add WebSocket/ROS communication
* [ ] Deploy inference on edge devices
* [ ] Add real physical bionic hand integration

---

# ⚠️ Limitations

The current system has several practical limitations:

* Orientation is estimated from MediaPipe landmark geometry rather than a dedicated IMU.
* Accuracy depends on lighting, camera quality, hand visibility, and landmark stability.
* The current implementation supports one detected hand.
* Gesture recognition is primarily pose-based rather than explicitly temporal.
* The orientation-impact experiment requires a separate orientation-free baseline model for a direct comparison.
* Dataset quality and orientation diversity strongly influence generalization.
* Random Forest predictions depend on the distribution represented in the training dataset.

---

# 👨‍💻 Author

**Arun M**

Robotics & Automation | Physical AI | Computer Vision | Robot Learning

GitHub:
https://github.com/dreamerarun

---

# ⭐ Project Summary

This project explores a more orientation-aware approach to vision-based bionic hand control.

Instead of using only static hand landmarks, the system combines **hand geometry with yaw, pitch, and roll** and feeds these features directly into a machine-learning classifier.

The result is a complete pipeline for:

```text
Hand Tracking
      +
Feature Engineering
      +
Orientation Estimation
      +
Machine Learning
      +
Real-Time Prediction
      +
Virtual Bionic Hand Visualization
```

The project provides a foundation for experimenting with **vision-based human-machine interfaces, gesture-controlled robotics, and physical AI applications**.
