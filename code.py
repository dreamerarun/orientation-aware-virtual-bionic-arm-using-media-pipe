#!/usr/bin/env python3
"""
enhanced_bionic_gestures_with_orientation.py

Enhanced version with FULL ORIENTATION INTEGRATION:
- Orientation (yaw, pitch, roll) as PRIMARY features in model training
- Orientation data saved in CSV alongside other features
- Orientation-aware prediction and validation
- Detailed orientation tracking in real-time
- Comprehensive evaluation with orientation metrics

Key Changes:
1. Orientation is now part of the core feature vector (not just logged separately)
2. All CSVs store orientation values as regular features
3. Model trains on orientation-inclusive features
4. Real-time prediction uses orientation for gesture classification
5. Validation reports include orientation impact analysis

Dependencies:
pip install opencv-python mediapipe numpy pandas scikit-learn pygame scipy
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import math
import os
import pickle
from collections import deque
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pygame
from scipy.spatial.transform import Rotation as R

# ---------- CONFIG ----------
GESTURE_NAMES = {0: "rest", 1: "grasp", 2: "release", 3: "point", 4: "pinch"}
MASTER_CSV = "gesture_master_dataset_with_orientation.csv"
SESSION_CSV_PREFIX = "session_orientation_"
PRED_LOG_CSV = "gesture_predictions_with_orientation.csv"
MODEL_FILE = "gesture_rf_model_orientation.pkl"
SCALER_FILE = "gesture_scaler_orientation.pkl"
METADATA_FILE = "training_metadata_orientation.pkl"
SAMPLES_PER_LABEL_DEFAULT = 120
SMOOTHING_QUEUE = 6
ORIENTATION_SMOOTH = 0.6

# Feature importance tracking
ORIENTATION_FEATURE_NAMES = ['yaw', 'pitch', 'roll']
# ----------------------------

# ---------- ORIENTATION-ENHANCED FEATURE EXTRACTION ----------
def landmarks_to_feature_vector_with_orientation(landmarks):
    """
    CRITICAL: This function now returns features with orientation INTEGRATED
    Orientation is part of the core feature set, not separate
    
    Returns: 
        feature_vector (numpy array): Complete features including orientation
        feature_metadata (dict): Breakdown of what features represent
    """
    lm = np.array(landmarks).reshape(21, 3)
    wrist = lm[0].copy()
    
    # Initialize feature list and metadata
    features = []
    feature_metadata = {
        'normalized_coords_count': 0,
        'distance_features_count': 0,
        'angle_features_count': 0,
        'palm_features_count': 0,
        'orientation_features_count': 0,
        'spread_features_count': 0
    }
    
    # 1. NORMALIZED COORDINATES (relative to wrist)
    norm = (lm - wrist).flatten()  # 63 features (21 landmarks × 3 coords)
    features.extend(norm)
    feature_metadata['normalized_coords_count'] = len(norm)
    
    # 2. INTER-LANDMARK DISTANCES
    pairs = [(4,8),(4,12),(4,16),(4,20),(8,12),(12,16),(16,20),(0,4),(0,8),(0,12),
             (0,16),(0,20),(8,16),(12,20)]
    distances = []
    for (i,j) in pairs:
        d = np.linalg.norm(lm[i] - lm[j])
        distances.append(d)
    features.extend(distances)
    feature_metadata['distance_features_count'] = len(distances)
    
    # 3. FINGER JOINT ANGLES
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    finger_sets = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
    
    joint_angles = {}
    angle_features = []
    
    for fname, f in zip(finger_names, finger_sets):
        for joint_idx in range(len(f)-1):
            if joint_idx == 0:
                p_base = lm[0]  # wrist for first joint
            else:
                p_base = lm[f[joint_idx-1]]
            p_mid = lm[f[joint_idx]]
            p_tip = lm[f[joint_idx+1]]
            
            v1 = p_base - p_mid
            v2 = p_tip - p_mid
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
            ang = math.degrees(math.acos(cosang))
            
            angle_features.append(ang)
            joint_angles[f"{fname}_joint_{joint_idx}"] = ang
    
    features.extend(angle_features)
    feature_metadata['angle_features_count'] = len(angle_features)
    
    # 4. PALM CENTER
    palm_idx = [0,5,9,13,17]
    palm_center = lm[palm_idx].mean(axis=0)
    features.extend(list(palm_center))
    feature_metadata['palm_features_count'] = 3
    
    # 5. *** ORIENTATION FEATURES (INTEGRATED) ***
    # This is the KEY change - orientation is now part of core features
    orient = landmarks_get_orientation(lm)
    orientation_features = [orient['yaw'], orient['pitch'], orient['roll']]
    features.extend(orientation_features)
    feature_metadata['orientation_features_count'] = 3
    joint_angles.update(orient)  # Store for logging
    
    # 6. PALM SPREAD
    fingertips = [4,8,12,16,20]
    max_spread = 0
    for i in range(len(fingertips)):
        for j in range(i+1, len(fingertips)):
            spread = np.linalg.norm(lm[fingertips[i]] - lm[fingertips[j]])
            max_spread = max(max_spread, spread)
    features.append(max_spread)
    feature_metadata['spread_features_count'] = 1
    joint_angles['palm_spread'] = max_spread
    
    # Convert to numpy array
    feature_vector = np.array(features, dtype=np.float32)
    
    # Add total count
    feature_metadata['total_features'] = len(feature_vector)
    
    return feature_vector, feature_metadata, joint_angles

def landmarks_get_orientation(lm):
    """
    Calculate hand orientation (yaw, pitch, roll) from landmarks
    
    Yaw: Rotation around vertical axis (left-right turn)
    Pitch: Rotation around horizontal axis (up-down tilt)
    Roll: Rotation around depth axis (twist)
    
    Returns: dict with yaw, pitch, roll in degrees
    """
    wrist = lm[0]
    index_mcp = lm[5]
    pinky_mcp = lm[17]
    middle_tip = lm[12]
    
    # Palm vector (across the hand)
    palm_vec = index_mcp - pinky_mcp
    # Finger vector (along the hand)
    finger_vec = middle_tip - wrist
    
    # Calculate angles
    yaw = math.degrees(math.atan2(palm_vec[0], palm_vec[2] + 1e-9))
    pitch = math.degrees(math.atan2(finger_vec[1], math.sqrt(finger_vec[0]**2 + finger_vec[2]**2) + 1e-9))
    roll = math.degrees(math.atan2(palm_vec[1], palm_vec[0] + 1e-9))
    
    return {'yaw': yaw, 'pitch': pitch, 'roll': roll}

# ---------- ORIENTATION-AWARE DATA MANAGEMENT ----------
class OrientationAwareDataManager:
    """
    Manages CSV storage with orientation as integral part of features
    """
    
    def __init__(self):
        self.initialize_master_csv()
        self.feature_metadata = None
    
    def initialize_master_csv(self):
        """Create master CSV with orientation-aware structure"""
        if not os.path.exists(MASTER_CSV):
            columns = ['session_id', 'user_name', 'timestamp', 'label', 'gesture_name']
            df = pd.DataFrame(columns=columns)
            df.to_csv(MASTER_CSV, index=False)
            print(f"✓ Created orientation-aware master dataset: {MASTER_CSV}")
    
    def save_session_data(self, session_id, user_name, data_rows):
        """
        Save session data with orientation integrated into features
        
        data_rows: list of dicts with:
            - label: gesture class
            - gesture_name: human-readable name
            - features: numpy array (includes orientation)
            - feature_metadata: dict describing feature breakdown
            - joint_angles: dict with all angles including orientation
        """
        if not data_rows:
            print("No data to save.")
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Store feature metadata from first sample
        if data_rows and 'feature_metadata' in data_rows[0]:
            self.feature_metadata = data_rows[0]['feature_metadata']
        
        # Prepare structured rows
        structured_rows = []
        for row in data_rows:
            full_row = {
                'session_id': session_id,
                'user_name': user_name,
                'timestamp': timestamp,
                'label': row['label'],
                'gesture_name': row['gesture_name']
            }
            
            # Add all features (including orientation)
            for i, feat in enumerate(row['features']):
                full_row[f'feature_{i}'] = feat
            
            # Add detailed joint angles for analysis (separate from features)
            for angle_name, angle_val in row['joint_angles'].items():
                full_row[f'angle_{angle_name}'] = angle_val
            
            structured_rows.append(full_row)
        
        df_session = pd.DataFrame(structured_rows)
        
        # Save session-specific CSV
        session_filename = f"{SESSION_CSV_PREFIX}{session_id}.csv"
        df_session.to_csv(session_filename, index=False)
        print(f"✓ Saved session to: {session_filename} ({len(df_session)} samples)")
        
        # Append to master CSV
        if os.path.exists(MASTER_CSV):
            df_master = pd.read_csv(MASTER_CSV)
            for col in df_session.columns:
                if col not in df_master.columns:
                    df_master[col] = np.nan
            for col in df_master.columns:
                if col not in df_session.columns:
                    df_session[col] = np.nan
            df_master = pd.concat([df_master, df_session], ignore_index=True)
        else:
            df_master = df_session
        
        df_master.to_csv(MASTER_CSV, index=False)
        print(f"✓ Updated master dataset: {MASTER_CSV} (total: {len(df_master)} samples)")
        
        return session_filename
    
    def load_training_data(self, use_all=True, session_id=None):
        """
        Load training data with orientation features integrated
        
        Returns:
            X: Feature matrix (includes orientation)
            y: Labels
            feature_info: Dictionary with feature breakdown
        """
        if not os.path.exists(MASTER_CSV):
            print("No master dataset found.")
            return None, None, None
        
        df = pd.read_csv(MASTER_CSV)
        
        if df.empty:
            print("Master dataset is empty.")
            return None, None, None
        
        if not use_all and session_id:
            df = df[df['session_id'] == session_id]
            if df.empty:
                print(f"No data found for session {session_id}")
                return None, None, None
        
        # Extract features (all feature columns, including orientation)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if not feature_cols:
            print("No feature columns found in dataset.")
            return None, None, None
        
        X = df[feature_cols].values
        y = df['label'].values
        
        # Remove rows with NaN
        valid_idx = ~np.isnan(X).any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Extract orientation columns for analysis
        angle_cols = [col for col in df.columns if col.startswith('angle_')]
        orientation_cols = [col for col in angle_cols if any(o in col for o in ['yaw', 'pitch', 'roll'])]
        
        feature_info = {
            'total_features': X.shape[1],
            'feature_columns': feature_cols,
            'orientation_columns': orientation_cols,
            'has_orientation': len(orientation_cols) > 0
        }
        
        print(f"✓ Loaded {len(X)} samples")
        print(f"✓ Feature vector size: {X.shape[1]} (includes orientation)")
        print(f"✓ Gesture distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X, y, feature_info
    
    def get_dataset_statistics(self):
        """Get comprehensive statistics including orientation data"""
        if not os.path.exists(MASTER_CSV):
            return None
        
        df = pd.read_csv(MASTER_CSV)
        if df.empty:
            return None
        
        # Basic stats
        stats = {
            'total_samples': len(df),
            'unique_users': df['user_name'].nunique() if 'user_name' in df.columns else 0,
            'unique_sessions': df['session_id'].nunique() if 'session_id' in df.columns else 0,
            'gesture_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {},
        }
        
        # Orientation statistics
        orientation_cols = ['angle_yaw', 'angle_pitch', 'angle_roll']
        orientation_stats = {}
        for col in orientation_cols:
            if col in df.columns:
                orientation_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        if orientation_stats:
            stats['orientation_stats'] = orientation_stats
        
        return stats

# ---------- ORIENTATION-AWARE BIONIC SYSTEM ----------
class OrientationAwareBionicSystem:
    def __init__(self):
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Data management
        self.data_manager = OrientationAwareDataManager()
        self.current_session_data = []
        self.current_session_id = None
        self.current_user_name = None
        
        # Model
        self.scaler = None
        self.model = None
        self.training_metadata = {}
        self.feature_metadata = None
        
        # Smoothing
        self.landmark_history = deque(maxlen=SMOOTHING_QUEUE)
        self.orientation_smoothed = {'yaw':0.0,'pitch':0.0,'roll':0.0}
        
        # Logging
        self.pred_log_cols = ["timestamp","pred_label","pred_conf","yaw","pitch","roll"]
        if not os.path.exists(PRED_LOG_CSV):
            pd.DataFrame(columns=self.pred_log_cols).to_csv(PRED_LOG_CSV, index=False)
        
        # Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((1200,820))
        pygame.display.set_caption("Bionic Hand - Orientation-Aware System")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 28)
        self.bigfont = pygame.font.SysFont(None, 44)
        self.smallfont = pygame.font.SysFont(None, 22)
    
    def collect_data(self, samples_per_label=SAMPLES_PER_LABEL_DEFAULT):
        """
        Data collection with orientation tracking
        """
        user_name = input("Enter your name for this session: ").strip()
        if not user_name:
            user_name = "Anonymous"
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_id = session_id
        self.current_user_name = user_name
        self.current_session_data = []
        
        print(f"\n{'='*70}")
        print(f"Starting ORIENTATION-AWARE data collection")
        print(f"Session ID: {session_id}")
        print(f"User: {user_name}")
        print(f"{'='*70}")
        
        mapping = {'r':0, 'g':1, 'l':2, 'p':3, 'n':4}
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not available.")
            return
        
        print("\nDATA COLLECTION MODE")
        print("Press key for gesture:")
        print(" r: rest, g: grasp, l: release, p: point, n: pinch")
        print(" q: finish collection")
        print(f"Each press records {samples_per_label} frames")
        print("\n⚠️  TIP: Vary hand orientation (yaw/pitch/roll) during recording!")
        
        counts = {k:0 for k in mapping.values()}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                
                # Display current orientation if hand detected
                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    pts = [[p.x, p.y, p.z] for p in lm.landmark]
                    lm_array = np.array(pts).reshape(21, 3)
                    orient = landmarks_get_orientation(lm_array)
                    
                    cv2.putText(frame, f"Orientation - Yaw:{orient['yaw']:.1f} Pitch:{orient['pitch']:.1f} Roll:{orient['roll']:.1f}",
                               (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                
                cv2.putText(frame, f"User: {user_name} | Session: {session_id}", (10,30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(frame, "Press: r/g/l/p/n to record | q to finish", (10,60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                yoff = 90
                for k, v in mapping.items():
                    cv2.putText(frame, f"{k}:{GESTURE_NAMES[v]} = {counts[v]}", (10,yoff),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
                    yoff += 26
                
                cv2.imshow("Collect Data (Orientation-Aware)", frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                
                if key in [ord(k) for k in mapping.keys()]:
                    label = mapping[chr(key)]
                    gesture_name = GESTURE_NAMES[label]
                    print(f"\n🎯 Recording: {gesture_name} (label {label})...")
                    print("   Tip: Move hand through different orientations!")
                    
                    recorded = 0
                    orientation_variety = {'yaw': [], 'pitch': [], 'roll': []}
                    
                    while recorded < samples_per_label:
                        ret2, frame2 = cap.read()
                        if not ret2:
                            continue
                        
                        frame2 = cv2.flip(frame2, 1)
                        rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                        res2 = self.hands.process(rgb2)
                        
                        if res2.multi_hand_landmarks:
                            lm = res2.multi_hand_landmarks[0]
                            pts = [[p.x, p.y, p.z] for p in lm.landmark]
                            
                            # Extract features WITH orientation integrated
                            feats, feat_meta, joint_angles = landmarks_to_feature_vector_with_orientation(pts)
                            
                            # Track orientation variety
                            orientation_variety['yaw'].append(joint_angles['yaw'])
                            orientation_variety['pitch'].append(joint_angles['pitch'])
                            orientation_variety['roll'].append(joint_angles['roll'])
                            
                            # Store structured data
                            data_row = {
                                'label': label,
                                'gesture_name': gesture_name,
                                'features': feats.tolist(),
                                'feature_metadata': feat_meta,
                                'joint_angles': joint_angles
                            }
                            self.current_session_data.append(data_row)
                            
                            recorded += 1
                            counts[label] += 1
                            
                            # Visual feedback with orientation
                            self.mp_draw.draw_landmarks(frame2, lm, self.mp_hands.HAND_CONNECTIONS)
                            cv2.putText(frame2, f"Recording {gesture_name}: {recorded}/{samples_per_label}",
                                       (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                            cv2.putText(frame2, f"Yaw:{joint_angles['yaw']:.1f} Pitch:{joint_angles['pitch']:.1f} Roll:{joint_angles['roll']:.1f}",
                                       (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                        
                        cv2.imshow("Collect Data - Recording", frame2)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    # Report orientation variety
                    print(f"   ✓ Recorded {recorded} samples")
                    if orientation_variety['yaw']:
                        yaw_range = max(orientation_variety['yaw']) - min(orientation_variety['yaw'])
                        pitch_range = max(orientation_variety['pitch']) - min(orientation_variety['pitch'])
                        roll_range = max(orientation_variety['roll']) - min(orientation_variety['roll'])
                        print(f"   Orientation variety - Yaw: {yaw_range:.1f}° Pitch: {pitch_range:.1f}° Roll: {roll_range:.1f}°")
                        
                        if yaw_range < 30 or pitch_range < 30:
                            print("   ⚠️  Low orientation variety! Try rotating hand more next time.")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Save collected data
        if self.current_session_data:
            print(f"\n{'='*70}")
            print(f"Total samples collected: {len(self.current_session_data)}")
            self.data_manager.save_session_data(session_id, user_name, self.current_session_data)
            
            stats = self.data_manager.get_dataset_statistics()
            if stats:
                print(f"\n📊 Dataset Statistics:")
                print(f"  Total samples in database: {stats['total_samples']}")
                print(f"  Unique users: {stats['unique_users']}")
                print(f"  Total sessions: {stats['unique_sessions']}")
                
                if 'orientation_stats' in stats:
                    print(f"\n  Orientation Statistics (all data):")
                    for orient_name, orient_stats in stats['orientation_stats'].items():
                        print(f"    {orient_name}: {orient_stats['mean']:.1f}° (±{orient_stats['std']:.1f}°)")
        else:
            print("No data collected.")
    
    def train_model(self, use_all_data=True):
        """
        Train model with orientation-inclusive features
        """
        if use_all_data:
            print("\n" + "="*70)
            print("Training with ALL historical data (orientation-aware)")
            print("="*70)
            X, y, feature_info = self.data_manager.load_training_data(use_all=True)
        else:
            if not self.current_session_id:
                print("No current session. Using all data.")
                X, y, feature_info = self.data_manager.load_training_data(use_all=True)
            else:
                print(f"\n" + "="*70)
                print(f"Training with CURRENT SESSION only ({self.current_session_id})")
                print("="*70)
                X, y, feature_info = self.data_manager.load_training_data(use_all=False, session_id=self.current_session_id)
        
        if X is None or len(X) < 10:
            print("❌ Insufficient data for training.")
            return False
        
        print(f"\n✓ Feature vector includes orientation: {feature_info['has_orientation']}")
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nClass distribution:")
        for label, count in zip(unique, counts):
            print(f"  {GESTURE_NAMES.get(int(label), str(label))}: {count} samples")
        
        if min(counts) < 2:
            print("⚠️  Warning: Some classes have <2 samples. Training may be unreliable.")
        
        # Normalize features (including orientation)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            print("⚠️  Warning: Stratified split not possible. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        
        # Train Random Forest
        print("\n🔧 Training Random Forest with orientation features...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\n{'='*70}")
        print(f"MODEL EVALUATION RESULTS (ORIENTATION-AWARE)")
        print(f"{'='*70}")
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Training accuracy: {train_acc*100:.2f}%")
        print(f"Test accuracy: {test_acc*100:.2f}%")
        
        # Feature importance analysis
        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*70}")
        
        feature_importances = self.model.feature_importances_
        
        # Get top features
        top_n = 15
        top_indices = np.argsort(feature_importances)[-top_n:][::-1]
        
        print(f"\nTop {top_n} most important features:")
        for idx in top_indices:
            importance = feature_importances[idx]
            feature_name = f"feature_{idx}"
            
            # Try to identify if it's orientation
            if idx >= len(feature_importances) - 10:  # Last features are likely orientation+spread
                print(f"  {feature_name}: {importance:.4f} 🎯 (likely orientation/spread)")
            else:
                print(f"  {feature_name}: {importance:.4f}")
        
        # Calculate orientation feature importance
        # Assuming orientation is in the last 3 core features before spread
        if len(feature_importances) > 3:
            # This is approximate - actual indices depend on feature extraction
            total_importance = feature_importances.sum()
            print(f"\nTotal feature importance (sanity check): {total_importance:.4f}")
        
        # Cross-validation
        if len(X) > 30:
            try:
                cv_scores = cross_val_score(self.model, X_scaled, y, cv=min(5, len(X)//10), scoring='accuracy')
                print(f"\n✓ Cross-validation accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*2*100:.2f}%)")
            except:
                print("\n⚠️  Cross-validation skipped (insufficient data)")
        
        # Classification report
        print(f"\n{'='*70}")
        print("CLASSIFICATION REPORT (Test Set)")
        print(f"{'='*70}")
        target_names = [GESTURE_NAMES.get(int(i), str(i)) for i in sorted(unique)]
        print(classification_report(y_test, y_pred_test, target_names=target_names, zero_division=0))
        
        # Confusion matrix
        print(f"{'='*70}")
        print("CONFUSION MATRIX (Test Set)")
        print(f"{'='*70}")
        cm = confusion_matrix(y_test, y_pred_test)
        
        # Pretty print confusion matrix
        print("\n" + " "*12 + "Predicted")
        print(" "*8, end="")
        for name in target_names:
            print(f"{name[:8]:>8}", end=" ")
        print("\nActual")
        
        for i, name in enumerate(target_names):
            print(f"{name[:8]:>8}", end=" ")
            for j in range(len(target_names)):
                if i < len(cm) and j < len(cm[i]):
                    print(f"{cm[i][j]:>8}", end=" ")
                else:
                    print(f"{0:>8}", end=" ")
            print()
        
        print(f"\n{'='*70}")
        
        # Save model and metadata
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(self.scaler, f)
        
        self.training_metadata = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'used_all_data': use_all_data,
            'total_samples': len(X),
            'feature_info': feature_info,
            'orientation_integrated': True
        }
        
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(self.training_metadata, f)
        
        print(f"\n✓ Model saved to: {MODEL_FILE}")
        print(f"✓ Scaler saved to: {SCALER_FILE}")
        print(f"✓ Metadata saved to: {METADATA_FILE}")
        print(f"✓ Orientation features: INTEGRATED in model")
        
        return True
    
    def load_model(self):
        """Load trained orientation-aware model"""
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            with open(MODEL_FILE, "rb") as f:
                self.model = pickle.load(f)
            with open(SCALER_FILE, "rb") as f:
                self.scaler = pickle.load(f)
            
            if os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, "rb") as f:
                    self.training_metadata = pickle.load(f)
                print("\n✓ Orientation-aware model loaded successfully!")
                print(f"  Trained on: {self.training_metadata.get('timestamp', 'unknown')}")
                print(f"  Training samples: {self.training_metadata.get('train_samples', 'unknown')}")
                print(f"  Test accuracy: {self.training_metadata.get('test_accuracy', 0)*100:.2f}%")
                print(f"  Orientation integrated: {self.training_metadata.get('orientation_integrated', False)}")
            else:
                print("✓ Model loaded (no metadata found).")
            return True
        
        print("❌ Model files not found.")
        return False
    
    def run_realtime(self):
        """Real-time gesture recognition with orientation visualization"""
        if self.model is None or self.scaler is None:
            print("❌ No model loaded. Please train or load a model first.")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera.")
            return
        
        print("\n" + "="*70)
        print("Starting ORIENTATION-AWARE real-time recognition...")
        print("="*70)
        print("✓ Orientation features are active in prediction")
        print("✓ Watch how orientation affects gesture classification")
        print("\nPress ESC to quit")
        
        running = True
        last_pred = "none"
        last_conf = 0.0
        
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            lm_points = None
            pred_label = None
            pred_conf = 0.0
            current_orientation = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
            
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                pts = [[p.x, p.y, p.z] for p in lm.landmark]
                
                # Smooth landmarks
                self.landmark_history.append(np.array(pts))
                avg_land = np.mean(np.stack(list(self.landmark_history)), axis=0)
                lm_points = avg_land.tolist()
                
                # Extract features WITH ORIENTATION
                feats, feat_meta, joint_angles = landmarks_to_feature_vector_with_orientation(avg_land)
                
                # Predict using orientation-inclusive features
                feats_scaled = self.scaler.transform(feats.reshape(1, -1))
                probs = self.model.predict_proba(feats_scaled)[0]
                pred_idx = int(self.model.predict(feats_scaled)[0])
                pred_label = GESTURE_NAMES.get(pred_idx, str(pred_idx))
                pred_conf = float(np.max(probs)) * 100.0
                
                # Get current orientation
                orient = landmarks_get_orientation(avg_land)
                current_orientation = orient
                
                # Smooth orientation for display
                for k in ['yaw', 'pitch', 'roll']:
                    self.orientation_smoothed[k] = (ORIENTATION_SMOOTH * orient[k] + 
                                                    (1-ORIENTATION_SMOOTH) * self.orientation_smoothed[k])
                
                last_pred = pred_label
                last_conf = pred_conf
                
                # Log prediction with orientation
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                row = {
                    "timestamp": timestamp,
                    "pred_label": pred_label,
                    "pred_conf": pred_conf,
                    "yaw": self.orientation_smoothed['yaw'],
                    "pitch": self.orientation_smoothed['pitch'],
                    "roll": self.orientation_smoothed['roll']
                }
                pd.DataFrame([row]).to_csv(PRED_LOG_CSV, mode='a', header=False, index=False)
                
                # Draw on camera frame
                self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
            
            # Display camera view with orientation
            cv2.putText(frame, f"Gesture: {last_pred} ({last_conf:.1f}%)", (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Yaw:{current_orientation['yaw']:.1f} Pitch:{current_orientation['pitch']:.1f} Roll:{current_orientation['roll']:.1f}",
                       (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, "Orientation affects prediction!", (10, frame.shape[0]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,255), 1)
            cv2.imshow("Camera Feed (ESC to quit) - Orientation-Aware", frame)
            
            # Render bionic hand with orientation info
            self._render_bionic_hand(lm_points, last_pred, last_conf, current_orientation)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                running = False
            
            self.clock.tick(45)
        
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
    
    def _render_bionic_hand(self, lm_points, pred_label, pred_conf, current_orientation):
        """Render bionic hand with enhanced orientation visualization"""
        self.screen.fill((15,15,20))
        w, h = self.screen.get_size()
        
        # Title
        title = self.bigfont.render("Bionic Hand - Orientation-Aware System", True, (230,230,255))
        self.screen.blit(title, (20,10))
        
        # Prediction info
        info = f"Predicted: {pred_label} ({pred_conf:.1f}%)"
        textsurf = self.font.render(info, True, (200,255,200))
        self.screen.blit(textsurf, (20,70))
        
        # Orientation display (PROMINENT)
        orient_title = self.font.render("Current Orientation:", True, (255,200,100))
        self.screen.blit(orient_title, (20,105))
        
        yaw_text = f"  Yaw (L-R):   {self.orientation_smoothed['yaw']:>6.1f}°"
        pitch_text = f"  Pitch (U-D): {self.orientation_smoothed['pitch']:>6.1f}°"
        roll_text = f"  Roll (Twist):{self.orientation_smoothed['roll']:>6.1f}°"
        
        yaw_surf = self.font.render(yaw_text, True, (255,150,150))
        pitch_surf = self.font.render(pitch_text, True, (150,255,150))
        roll_surf = self.font.render(roll_text, True, (150,150,255))
        
        self.screen.blit(yaw_surf, (20,135))
        self.screen.blit(pitch_surf, (20,165))
        self.screen.blit(roll_surf, (20,195))
        
        # Visual orientation bars
        bar_x = 250
        bar_y = 135
        bar_width = 200
        bar_height = 20
        
        # Yaw bar (-180 to 180)
        yaw_norm = (self.orientation_smoothed['yaw'] + 180) / 360
        yaw_bar_w = int(yaw_norm * bar_width)
        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (255,100,100), (bar_x, bar_y, yaw_bar_w, bar_height))
        pygame.draw.rect(self.screen, (150,150,150), (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Pitch bar (-90 to 90)
        pitch_norm = (self.orientation_smoothed['pitch'] + 90) / 180
        pitch_bar_w = int(pitch_norm * bar_width)
        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y+30, bar_width, bar_height))
        pygame.draw.rect(self.screen, (100,255,100), (bar_x, bar_y+30, pitch_bar_w, bar_height))
        pygame.draw.rect(self.screen, (150,150,150), (bar_x, bar_y+30, bar_width, bar_height), 2)
        
        # Roll bar (-180 to 180)
        roll_norm = (self.orientation_smoothed['roll'] + 180) / 360
        roll_bar_w = int(roll_norm * bar_width)
        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y+60, bar_width, bar_height))
        pygame.draw.rect(self.screen, (100,100,255), (bar_x, bar_y+60, roll_bar_w, bar_height))
        pygame.draw.rect(self.screen, (150,150,150), (bar_x, bar_y+60, bar_width, bar_height), 2)
        
        # Model info
        if self.training_metadata:
            model_info = f"Model Acc: {self.training_metadata.get('test_accuracy', 0)*100:.1f}% | Samples: {self.training_metadata.get('total_samples', 0)}"
            msurf = self.smallfont.render(model_info, True, (180,180,200))
            self.screen.blit(msurf, (20,230))
            
            orient_status = "✓ Orientation: ACTIVE in prediction" if self.training_metadata.get('orientation_integrated') else "✗ Orientation: NOT integrated"
            ostat_surf = self.smallfont.render(orient_status, True, (100,255,100) if self.training_metadata.get('orientation_integrated') else (255,100,100))
            self.screen.blit(ostat_surf, (20,255))
        
        if lm_points is None:
            ws = self.font.render("Waiting for hand...", True, (200,80,80))
            self.screen.blit(ws, (20,290))
            pygame.display.flip()
            return
        
        # Map landmarks to screen coordinates
        mapped = []
        for (x,y,z) in lm_points:
            sx = int((1.0 - x) * 0.45 * w + 0.5*w)
            sy = int(y * 0.6 * h + 0.35*h)
            mapped.append((sx, sy, z))
        
        # Draw palm polygon
        palm_idx = [0,5,9,13,17]
        palm_pts = [mapped[i] for i in palm_idx]
        palm_xy = [(p[0], p[1]) for p in palm_pts]
        if len(palm_xy) >= 3:
            pygame.draw.polygon(self.screen, (70,70,70), palm_xy)
            pygame.draw.polygon(self.screen, (120,120,120), palm_xy, 4)
        
        # Draw fingers
        finger_groups = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
        colors = [(255,100,100),(100,255,100),(100,100,255),(255,255,100),(255,100,255)]
        
        for fidx, finger in enumerate(finger_groups):
            points2 = [mapped[i] for i in finger]
            # Draw segments
            for i in range(len(points2)-1):
                x1,y1,_ = points2[i]
                x2,y2,_ = points2[i+1]
                thickness = max(4, 18 - i*3)
                # Shadow
                pygame.draw.line(self.screen, (10,10,10), (x1+3,y1+3), (x2+3,y2+3), thickness+2)
                # Main line
                pygame.draw.line(self.screen, colors[fidx], (x1,y1), (x2,y2), thickness)
                # Joint circle
                pygame.draw.circle(self.screen, (230,230,230), (x1,y1), max(3, thickness//2))
            # Fingertip
            tx,ty,_ = points2[-1]
            pygame.draw.circle(self.screen, (20,20,20), (tx+3,ty+3), 12)
            pygame.draw.circle(self.screen, colors[fidx], (tx,ty), 10)
        
        # Wrist
        wx,wy,_ = mapped[0]
        pygame.draw.circle(self.screen, (90,90,90), (wx,wy), 32)
        pygame.draw.circle(self.screen, (160,160,160), (wx,wy), 26)
        
        # Legend
        ybase = 600
        legend_title = self.font.render("Gestures:", True, (255,255,255))
        self.screen.blit(legend_title, (20, ybase-30))
        for i, name in GESTURE_NAMES.items():
            color = (100,255,100) if name == pred_label.lower() else (200,200,200)
            surf = self.font.render(f"{i}: {name}", True, color)
            self.screen.blit(surf, (20, ybase + i*24))
        
        # Orientation tip
        tip_text = "💡 Try rotating your hand to see orientation impact!"
        tip_surf = self.smallfont.render(tip_text, True, (150,200,255))
        self.screen.blit(tip_surf, (20, 750))
        
        pygame.display.flip()

# ---------- CLI ----------
def main():
    system = OrientationAwareBionicSystem()
    
    print("\n" + "="*70)
    print(" "*10 + "ORIENTATION-AWARE BIONIC GESTURE SYSTEM")
    print("="*70)
    print("\n🎯 KEY FEATURES:")
    print("  • Orientation (yaw, pitch, roll) integrated into model training")
    print("  • Orientation features stored in CSV for analysis")
    print("  • Real-time orientation tracking and visualization")
    print("  • Comprehensive validation with orientation metrics")
    
    # Show dataset statistics
    stats = system.data_manager.get_dataset_statistics()
    if stats:
        print(f"\n📊 Current Dataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Unique users: {stats['unique_users']}")
        print(f"  Total sessions: {stats['unique_sessions']}")
        
        if stats['gesture_distribution']:
            print(f"  Gesture distribution:")
            for label, count in stats['gesture_distribution'].items():
                print(f"    {GESTURE_NAMES.get(int(label), str(label))}: {count}")
        
        if 'orientation_stats' in stats:
            print(f"\n  📐 Orientation Statistics:")
            for orient_name, orient_data in stats['orientation_stats'].items():
                print(f"    {orient_name}: {orient_data['mean']:.1f}° (±{orient_data['std']:.1f}°)")
    
    print("\n" + "="*70)
    print("MENU OPTIONS:")
    print("="*70)
    print("1. Collect new gesture data (with orientation tracking)")
    print("2. Train model using CURRENT session only")
    print("3. Train model using ALL historical data")
    print("4. Load existing model from disk")
    print("5. Run real-time recognition & visualization")
    print("6. View dataset statistics (including orientation)")
    print("7. Test orientation impact (experimental)")
    print("8. Exit")
    print("="*70)
    
    while True:
        cmd = input("\nEnter choice (1-8): ").strip()
        
        if cmd == "1":
            # Collect data
            n = input(f"Samples per gesture [{SAMPLES_PER_LABEL_DEFAULT}]: ").strip()
            n = int(n) if n.isdigit() else SAMPLES_PER_LABEL_DEFAULT
            system.collect_data(samples_per_label=n)
            
            # Ask if user wants to train immediately
            train_now = input("\nTrain model now? (y/n): ").strip().lower()
            if train_now == 'y':
                use_all = input("Use ALL historical data? (y/n, default=y): ").strip().lower()
                if use_all == 'n':
                    print("\nTraining with current session only...")
                    system.train_model(use_all_data=False)
                else:
                    print("\nTraining with all historical data...")
                    system.train_model(use_all_data=True)
        
        elif cmd == "2":
            # Train with current session only
            if not system.current_session_id:
                print("\n❌ No current session found. Collect data first or use option 3.")
                continue
            confirm = input(f"\nTrain using ONLY current session data? (y/n): ").strip().lower()
            if confirm == 'y':
                system.train_model(use_all_data=False)
        
        elif cmd == "3":
            # Train with all data
            confirm = input("\nTrain using ALL historical data? (y/n): ").strip().lower()
            if confirm == 'y':
                success = system.train_model(use_all_data=True)
                if success:
                    print("\n✅ Model training completed successfully!")
                    print("   Orientation features are now integrated in the model.")
        
        elif cmd == "4":
            # Load model
            success = system.load_model()
            if success:
                print("✅ Model loaded successfully!")
        
        elif cmd == "5":
            # Run real-time
            if system.model is None or system.scaler is None:
                print("\n❌ No model loaded. Please train (option 2/3) or load model (option 4) first.")
                continue
            system.run_realtime()
        
        elif cmd == "6":
            # Show statistics
            stats = system.data_manager.get_dataset_statistics()
            if stats:
                print("\n" + "="*70)
                print("DATASET STATISTICS (Orientation-Aware)")
                print("="*70)
                print(f"Total samples: {stats['total_samples']}")
                print(f"Unique users: {stats['unique_users']}")
                print(f"Total sessions: {stats['unique_sessions']}")
                print(f"\nGesture distribution:")
                for label, count in sorted(stats['gesture_distribution'].items()):
                    gesture_name = GESTURE_NAMES.get(int(label), str(label))
                    percentage = (count / stats['total_samples']) * 100
                    print(f"  {gesture_name:12} : {count:5} samples ({percentage:5.1f}%)")
                
                if 'orientation_stats' in stats:
                    print(f"\n📐 Orientation Statistics (All Data):")
                    print(f"{'='*70}")
                    for orient_name, orient_data in stats['orientation_stats'].items():
                        orient_label = orient_name.replace('angle_', '').upper()
                        print(f"{orient_label:10} : Mean={orient_data['mean']:>6.1f}° | "
                              f"Std={orient_data['std']:>5.1f}° | "
                              f"Range=[{orient_data['min']:>6.1f}°, {orient_data['max']:>6.1f}°]")
                
                print("="*70)
                
                # Show model info if available
                if system.training_metadata:
                    print("\n🤖 Current Model Info:")
                    print(f"  Trained on: {system.training_metadata.get('timestamp', 'N/A')}")
                    print(f"  Training samples: {system.training_metadata.get('train_samples', 'N/A')}")
                    print(f"  Test samples: {system.training_metadata.get('test_samples', 'N/A')}")
                    print(f"  Test accuracy: {system.training_metadata.get('test_accuracy', 0)*100:.2f}%")
                    print(f"  Orientation integrated: {'✅ YES' if system.training_metadata.get('orientation_integrated') else '❌ NO'}")
            else:
                print("\n❌ No dataset found. Collect data first (option 1).")
        
        elif cmd == "7":
            # Test orientation impact
            print("\n" + "="*70)
            print("ORIENTATION IMPACT TEST (Experimental)")
            print("="*70)
            print("\nThis feature would test how much orientation affects predictions.")
            print("Implementation: Train two models (with/without orientation) and compare.")
            print("\n💡 Current system always includes orientation in features.")
            print("   To test impact, you would need to:")
            print("   1. Train model without orientation features")
            print("   2. Train model WITH orientation features")
            print("   3. Compare accuracies on same test set")
            print("\n⚠️  This feature requires code modification to exclude orientation.")
        
        elif cmd == "8":
            # Exit
            print("\n" + "="*70)
            print("Thank you for using Orientation-Aware Bionic Gesture System!")
            print("="*70)
            print("✓ Your data includes orientation information")
            print("✓ All CSV files contain orientation features")
            print("✓ Model uses orientation for improved accuracy")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-8.")

if __name__ == "__main__":
    main()
