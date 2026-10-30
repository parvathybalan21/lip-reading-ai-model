# 👄 Lip Reading AI: Spatiotemporal Sentence Predictor

This project implements a state-of-the-art **Lip Reading deep learning system** that predicts spoken sentences directly from silent video recordings of mouth movements. Combining **3D Convolutional Neural Networks (3D CNNs)** and **Bidirectional Long Short-Term Memory (Bi-LSTM)** networks with **Connectionist Temporal Classification (CTC)** loss, this model achieves near-perfect sequence prediction without needing any audio.

An interactive **Gradio Web Interface** is included to allow users to drag-and-drop silent video files, instantly preview the video, and view decoded text transcriptions with high accuracy.

---

## 📋 Features
- **Deep Spatiotemporal Feature Extraction**: Captures both visual spatial features (lip contours) and temporal kinetics (movement over time) using 3D convolutions.
- **Sequential Context Decoders**: Bidirectional LSTMs parse mouth transitions forwards and backwards to infer phonetic and vocabulary connections.
- **CTC Sequence Alignment**: Learns the mapping from input frames to text transcriptions without requiring frame-by-frame text alignment labels.
- **Automated Video Preprocessing**: Dynamic cropping of the lip region of interest `[190:236, 80:220]`, automatic grayscale conversion, and batch z-score normalization.
- **Robust Web Interface**: Interactive user interface built with Gradio including automatic `ffmpeg`-based video transcoding for standard browser compatibility.
- **Cross-Platform Compatibility**: Code structure supports path handling on Windows, Linux, and macOS.

---

## 📁 Project Structure
```
├── data/
│   ├── s1/                        # Directory for video files (.mpg / .mp4)
│   │   ├── bbal7s.mpg
│   │   └── bbaf2n.mpg
│   └── alignments/
│       └── s1/                    # Transcript files containing phoneme alignments
│           ├── bbal7s.align
│           └── bbaf2n.align
├── models/
│   └── checkpoint                 # Pretrained weights checkpoint files
├── assets/
│   └── predictions_visualized.gif # Animated prediction demonstrations
├── app.py                         # Gradio Web Application deployment code
├── model_pipeline.py              # Model architecture definition and processing helpers
└── README.md                      # Detailed project documentation
```

---

## 📝 About the Dataset
The system is built and trained on the **GRID Corpus**, a multi-speaker database of high-quality audio and video recordings of simple sentences. 
- **Syntactic Structure**: Each sentence consists of a strict 6-word grammatical sequence: 
  `[command:4] + [color:4] + [preposition:4] + [letter:25] + [digit:10] + [adverb:4]`
- **Example**: *"Set blue by u two please"* or *"Bin green at v five soon"*.
- **Target Vocabulary**: 39 unique characters (lowercase English alphabet, space token, numbers 1-9, and basic punctuation marks).

---

## ⚙️ Model Pipeline & Architecture

The neural network architecture uses a serial configuration to read lip shapes and translate them into a sequence of characters:

```
Input Video Sequence (75, 46, 140, 1) 
          │
          ▼
 3D CNN Layer 1   ──►  128 filters (3x3x3), Pool (1x2x2) [Activation: ReLU]
          │
          ▼
 3D CNN Layer 2   ──►  256 filters (3x3x3), Pool (1x2x2) [Activation: ReLU]
          │
          ▼
 3D CNN Layer 3   ──►  75 filters (3x3x3), Pool (1x2x2)  [Activation: ReLU]
          │
          ▼
TimeDistributed   ──►  Flattens spatial dimensions into (75, 6375) vectors
          │
          ▼
Bi-LSTM Layer 1   ──►  128 cells per direction (Orthogonal Init), Dropout (0.5)
          │
          ▼
Bi-LSTM Layer 2   ──►  128 cells per direction (Orthogonal Init), Dropout (0.5)
          │
          ▼
 Dense Logits     ──►  41 classes (Vocabulary + Blank token) with Softmax activation
```

---

## 🏋️ Training Strategy
- **Loss Function**: Connectionist Temporal Classification (CTC) Batch Cost (`ctc_batch_cost`) is used to dynamically map varying length character outputs without explicit alignment label timings.
- **Optimization**: Legacy Adam Optimizer (`Adam` with learning rate `1e-4` or scheduled decay).
- **Learning Rate Scheduling**: Custom decay function used to shrink learning rate incrementally over training epochs to ensure smooth model parameter convergence.
- **Weight Initialization**: Layers are initialized using `Orthogonal` weights for RNN gates and `He Normal` weights for classification outputs to minimize vanishing gradient risks.

---

## 📊 Evaluation & Verification
Prediction accuracy is verified on test datasets using both greedy and beam-search CTC decoding. Our evaluation loop runs side-by-side predictions checking true annotations directly:
- **Ground Truth**: `set blue by u two please`
- **AI Prediction**: `set blue by u two please`

--- 

## 💻 Technologies Used
- **Deep Learning Framework**: TensorFlow 2.15+ & TF-Keras
- **Computer Vision**: OpenCV (cv2) for video frame slicing and array manipulation
- **Web Server Interface**: Gradio
- **Media Utilities**: FFmpeg & ImageIO (for GIF compilation & transcoding processing)
- **Development Platform**: Python 3.10+ (compatible with Windows, macOS, and Linux)

---

## 🚀 Setup & Execution Guide

### 1. Windows Setup (via Command Prompt/PowerShell)
- Install system requirements:
  ```cmd
  pip install tensorflow tf_keras opencv-python matplotlib imageio gdown gradio
  ```
- Install FFmpeg dependency:
  ```powershell
  winget install Gyan.FFmpeg
  ```
  *Ensure `ffmpeg` binary path is added to your Windows environment PATH variables.*
- Run Web App:
  ```bash
  python app.py
  ```

### 2. Linux / macOS Setup
- Install library requirements:
  ```bash
  pip install tensorflow tf_keras opencv-python matplotlib imageio gdown gradio
  ```
- Install FFmpeg utility:
  ```bash
  # macOS
  brew install ffmpeg
  
  # Ubuntu/Debian
  sudo apt-get update && sudo apt-get install ffmpeg
  ```
- Run Web App:
  ```bash
  python app.py
  ```

---

## 🔮 Future Improvements
1. **Dynamic Landmark Detection**: Incorporate MediaPipe or Dlib facial landmark detectors to dynamically track and crop the mouth area, even if the speaker moves their head.
2. **Attention-Based Transformers**: Upgrade recurrent Bi-LSTM blocks to a Spatiotemporal Transformer (e.g., TM-Seq2Seq) to support longer conversational recordings.
3. **Word Error Rate (WER) Metrics**: Implement word error rate tracking inside training callbacks to better optimize linguistic comprehension.

---

## ✍️ Author
**Parvathy B S**  
AI / Deep Learning Developer  

---

## 🤝 Acknowledgements
- The creators of the **GRID Corpus** for making their audio-visual speech database publicly accessible.
- The open-source community behind **Keras** and **TensorFlow** for providing powerful spatiotemporal neural network interfaces.