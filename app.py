# streamlit_app.py
import streamlit as st
import cv2
import math
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import tempfile
import os
import mediapipe as mp
from scipy.spatial.distance import euclidean

# Set page config
st.set_page_config(page_title="Squat Form Analyzer", page_icon="🏋️", layout="wide")
st.title("Injury Risk Analyzer")
st.markdown("Upload a video of your squat to receive an instant AI-powered analysis of your form and potential injury risk.")

# --- Load the trained model ---
@st.cache_resource
def load_model():
    """Load the pre-trained model from the models directory."""
    try:
        model_path = Path("models/squat_risk_classifier.joblib")
        if model_path.exists():
            return joblib.load(model_path)
        else:
            st.error("❌ Model not found! Please make sure you've trained the model first.")
            st.info("Run `03_training.ipynb` to train and save the model.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Feature Extraction Functions (MUST match your training!) ---
# Define the same landmark indices you used during feature extraction
# --- Feature Extraction Functions (MUST match your training!) ---
# Define the same landmark indices you used during feature extraction
MP_POSE_INDICES = {
    'NOSE': 0,
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29,
    'RIGHT_HEEL': 30
}

def calculate_angle(a, b, c):
    """Calculate the angle between three points (vectors ab and bc)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def calculate_torso_angle(shoulder, hip):
    """Calculate the approximate torso angle from vertical."""
    # Vector from hip to shoulder
    torso_vector = (shoulder[0] - hip[0], shoulder[1] - hip[1])
    # Vertical vector (pointing down)
    vertical_vector = (0, 1)
    # Calculate angle
    dot_product = torso_vector[0]*vertical_vector[0] + torso_vector[1]*vertical_vector[1]
    magnitude = math.sqrt(torso_vector[0]**2 + torso_vector[1]**2) * math.sqrt(vertical_vector[0]**2 + vertical_vector[1]**2)
    cosine_angle = dot_product / magnitude
    angle_rad = math.acos(cosine_angle)
    return math.degrees(angle_rad)

def extract_features_from_frame(landmarks):
    """
    Extract biomechanical features from a single frame's landmarks.
    This MUST be identical to the function used in 02_feature_extraction.ipynb.
    """
    features = []
    
    # Get average of left and right hips for a more stable torso calculation
    l_hip = landmarks[MP_POSE_INDICES['LEFT_HIP']]
    r_hip = landmarks[MP_POSE_INDICES['RIGHT_HIP']]
    mid_hip = ((l_hip[0] + r_hip[0])/2, (l_hip[1] + r_hip[1])/2)
    
    l_shoulder = landmarks[MP_POSE_INDICES['LEFT_SHOULDER']]
    r_shoulder = landmarks[MP_POSE_INDICES['RIGHT_SHOULDER']]
    mid_shoulder = ((l_shoulder[0] + r_shoulder[0])/2, (l_shoulder[1] + r_shoulder[1])/2)

    # 1. Torso Angle (from vertical) - EXACTLY as in your notebook
    features.append(calculate_torso_angle(mid_shoulder, mid_hip))
    
    # 2. Left Knee Angle (Hip-Knee-Ankle)
    features.append(calculate_angle(
        landmarks[MP_POSE_INDICES['LEFT_HIP']],
        landmarks[MP_POSE_INDICES['LEFT_KNEE']],
        landmarks[MP_POSE_INDICES['LEFT_ANKLE']]
    ))
    
    # 3. Right Knee Angle
    features.append(calculate_angle(
        landmarks[MP_POSE_INDICES['RIGHT_HIP']],
        landmarks[MP_POSE_INDICES['RIGHT_KNEE']],
        landmarks[MP_POSE_INDICES['RIGHT_ANKLE']]
    ))
    
    # 4. Knee Valgus (Approximation - distance between knees vs ankles)
    knee_distance = euclidean(
        (landmarks[MP_POSE_INDICES['LEFT_KNEE']][0], landmarks[MP_POSE_INDICES['LEFT_KNEE']][1]),
        (landmarks[MP_POSE_INDICES['RIGHT_KNEE']][0], landmarks[MP_POSE_INDICES['RIGHT_KNEE']][1])
    )
    ankle_distance = euclidean(
        (landmarks[MP_POSE_INDICES['LEFT_ANKLE']][0], landmarks[MP_POSE_INDICES['LEFT_ANKLE']][1]),
        (landmarks[MP_POSE_INDICES['RIGHT_ANKLE']][0], landmarks[MP_POSE_INDICES['RIGHT_ANKLE']][1])
    )
    # Avoid division by zero
    features.append(knee_distance / ankle_distance if ankle_distance > 0 else 1.0)

    # 5. Left Hip Flexion (Angle at the hip - Shoulder-Hip-Knee)
    features.append(calculate_angle(
        landmarks[MP_POSE_INDICES['LEFT_SHOULDER']],
        landmarks[MP_POSE_INDICES['LEFT_HIP']],
        landmarks[MP_POSE_INDICES['LEFT_KNEE']]
    ))
    
    # 6. Right Hip Flexion
    features.append(calculate_angle(
        landmarks[MP_POSE_INDICES['RIGHT_SHOULDER']],
        landmarks[MP_POSE_INDICES['RIGHT_HIP']],
        landmarks[MP_POSE_INDICES['RIGHT_KNEE']]
    ))

    return features
# --- Main Application ---
uploaded_file = st.file_uploader("**Choose a squat video file**", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    if model is None:
        st.error("Cannot analyze without a trained model. Please load a model first.")
    else:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name

        # Initialize variables for processing
        all_features = []
        frame_count = 0
        total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

        # Create placeholders for UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_placeholder = st.empty()

        status_text.text("🔄 Starting video analysis...")

        # Process video using MediaPipe
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            cap = cv2.VideoCapture(video_path)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 5th frame to save time
                if frame_count % 5 == 0:
                    # Resize frame for faster processing
                    frame = cv2.resize(frame, (640, 480))
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame with MediaPipe
                    results = pose.process(rgb_frame)
                    if results.pose_landmarks:
                        landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
                        features = extract_features_from_frame(landmarks)
                        all_features.append(features)
                
                frame_count += 1
                # Update progress
                if frame_count % 30 == 0:
                    progress = frame_count / total_frames
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"🔄 Processing... Frame {frame_count}/{total_frames}")

            cap.release()

        # Clean up temp file
        os.unlink(video_path)

        progress_bar.progress(1.0)
        
        # Analyze results
        if all_features:
            features_array = np.array(all_features)
            
            # Make predictions
            predictions = model.predict(features_array)
            risk_percentage = np.mean(predictions) * 100
            
            # Display results
            status_text.text("✅ Analysis Complete!")
            results_placeholder.empty()
            
            with results_placeholder.container():
                st.subheader("📊 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Overall Risk Score", f"{risk_percentage:.1f}%")
                
                with col2:
                    if risk_percentage > 66:
                        st.error("🚨 High Injury Risk")
                        st.caption("Most frames showed risky form patterns like knee valgus or excessive forward lean.")
                    elif risk_percentage > 33:
                        st.warning("⚠️ Moderate Injury Risk")
                        st.caption("Many frames showed suboptimal form. Consider reviewing your technique.")
                    else:
                        st.success("✅ Low Injury Risk")
                        st.caption("Good form was maintained throughout most of the movement.")
                
                with col3:
                    st.metric("Frames Analyzed", len(predictions))

                # Detailed analysis
                st.subheader("📈 Detailed Breakdown")
                
                tab1, tab2 = st.tabs(["Frame-by-Frame Analysis", "Statistics"])
                
                with tab1:
                    st.caption("Risk prediction for each analyzed frame (1 = Risky, 0 = Safe)")
                    chart_data = pd.DataFrame({'Risk Prediction': predictions})
                    st.bar_chart(chart_data, height=250)
                
                with tab2:
                    safe_frames = len(predictions) - np.sum(predictions)
                    risky_frames = np.sum(predictions)
                    
                    st.metric("Safe Frames", safe_frames)
                    st.metric("Risky Frames", risky_frames)
                    st.metric("Confidence", f"{model.predict_proba(features_array).max(axis=1).mean():.2%}")
        
        else:
            status_text.text("❌ Analysis Failed")
            st.error("""
            Could not detect human poses in this video. This might be because:
            - The person is not fully visible in the frame
            - The video quality is too low
            - The camera angle makes pose estimation difficult
            - The person is too far from the camera
            """)
            st.info("💡 Try a different video with clear visibility of the whole body during the squat.")

else:
    st.info("👆 Please upload a video file to begin analysis.")
    st.divider()
    st.subheader("ℹ️ How to get the best results:")
    st.markdown("""
    - **Film horizontally** with good lighting
    - Ensure your **whole body is visible** throughout the squat
    - **Side-angle** videos work best for form analysis
    - Avoid shaky camera movements
    - The analysis takes longer for longer videos
    """)

# Footer
st.divider()
st.caption("Built with MediaPipe, Scikit-learn, and Streamlit | Injury Risk AI Project")