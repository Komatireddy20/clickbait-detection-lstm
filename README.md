# Clickbait Detection Using Deep Learning (LSTM)

A binary text classification system that detects clickbait headlines using an LSTM-based deep learning model with word2vec embeddings.

## Problem Statement
Clickbait headlines on social media platforms mislead users into clicking low-quality content. This project builds an automated detection system using deep learning to classify headlines as clickbait or non-clickbait.

## Approach
- Built and trained an **LSTM (Long Short-Term Memory)** neural network for sequence-based text classification
- Used **word2vec embeddings** (Google's pre-trained model trained on 100 billion words) to convert headlines into 300-dimensional vector representations
- Modified LSTM architecture to give higher weight to the second half of headlines, where clickbait signals are more prominent
- Applied full **NLP preprocessing pipeline**: tokenization, padding, sequence encoding, stop word handling

## Dataset
- Source: [Kaggle Clickbait Dataset](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset)
- 32,000 headlines from WikiNews, New York Times, The Guardian, BuzzFeed, Upworthy, ViralNova, and others
- Balanced: 50% clickbait, 50% non-clickbait
- Binary labels: 1 = Clickbait, 0 = Non-Clickbait

## Model Architecture
- Embedding Layer (vocab size: 5000, embedding size: 32)
- LSTM Layer (128 units)
- Dense Output Layer (Sigmoid activation)
- Optimizer: Adam | Loss: Binary Crossentropy
- Trained for 20 epochs with batch size 512

## Results
| Metric | Score |
|--------|-------|
| Accuracy | High |
| Precision | High |
| Recall | High |
| Evaluation | Confusion Matrix + Classification Report |

## Repository Structure
clickbait-detection-lstm/
│
├── clickbait_detection.ipynb   # Main notebook with full pipeline
├── app.py                      # Application script
├── clickbait_data.csv          # Dataset (32,000 headlines)
├── model.h5                    # Saved trained model weights
├── README.md
├── LICENSE
└── .gitignore
## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## How to Run
1. Clone the repository
```bash
git clone https://github.com/Komatireddy20/clickbait-detection-lstm
cd clickbait-detection-lstm
```
2. Install dependencies
```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn
```
3. Open the notebook
```bash
jupyter notebook clickbait_detection.ipynb
```
4. Run all cells in order

## Team
Developed as part of B.Tech final year project at Vignana Bharathi Institute of Technology (2023-24).
Project Lead: Sravya Komati Reddy
