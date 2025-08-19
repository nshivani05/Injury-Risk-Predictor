# 🏋️ Injury Risk AI - Squat Form Analyzer

A powerful AI application that analyzes squat exercises in real-time using computer vision and machine learning to detect biomechanical patterns that could lead to injury.

## ⚠️ Important Note: Large Files Not Hosted

**This repository contains ONLY code and documentation.** Several critical large files cannot be hosted on GitHub due to size limitations:

- **🚫 CLIP Model File (`ViT-B-32.pt` - 338MB)**: OpenAI's vision model must be downloaded automatically on first run
- **🚫 Video Dataset**: Original squat videos (`*.mp4` files) are not included for privacy and size reasons
- **🚫 Processed Data**: Extracted keypoints and features (`*.npy` files) are not included
- **🚫 Trained Models**: Machine learning models (`*.joblib` files) are not included

## 📥 Manual Installation Required

To run this project, you must manually install and download components:

```bash
# 1. Clone this repository
git clone https://github.com/nshivani05/Injury-Risk-Predictor.git
cd Injury-Risk-Predictor

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install CLIP model (will auto-download on first run)
pip install git+https://github.com/openai/CLIP.git

# 4. Add your own video data to data/videos/ folder
#    - risk1.mp4, risk2.mp4, etc. (risky squats)
#    - safe1.mp4, safe2.mp4, etc. (safe squats)

# 5. Run the processing notebooks in order:
#    - 01_pose_extraction.ipynb
#    - 02_feature_extraction.ipynb  
#    - 03_training.ipynb

# 6. Finally run the app
streamlit run app.py
