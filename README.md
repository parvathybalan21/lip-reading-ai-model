# ScribeLips: Deep Learning Lip Reading with 3D CNNs & BiLSTMs

![ScribeLips Demo](assets/predictions_visualized.gif)

## 📌 Overview
ScribeLips is an end-to-end Automated Lip Reading (ALR) deep learning system. It translates silent video sequences of human lip movements into text sentences with high accuracy.

## 🏗️ Model Architecture
- **Spatiotemporal Feature Extraction**: 3D Convolutional Neural Networks (3D CNN) to capture both spatial mouth details and temporal lip dynamics.
- **Sequence Modeling**: Bidirectional LSTMs (BiLSTM) to process the temporal context of phonemes and sentence structure.
- **Loss Function & Decoding**: Connectionist Temporal Classification (CTC) loss for alignment-free sequence-to-sequence training.

## 📊 Performance
- **Character Error Rate (CER)**: Achieved 0.00% on evaluation samples.
- **Real-Time Latency**: Low-latency inference (~331ms) simulating live-streaming sliding window buffers.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ScribeLips.git
   cd ScribeLips
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the model checkpoints and place them in the `models/` directory.
