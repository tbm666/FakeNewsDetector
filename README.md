# Fake News Detector README

# Fake News Detector

A machine learning model to classify news articles as REAL or FAKE using TF-IDF and PassiveAggressiveClassifier.

## 📊 Overview

This project implements a fake news detection system that achieves over 90% accuracy in classifying news articles. The model uses TF-IDF for text vectorization and PassiveAggressiveClassifier for classification.

## 🚀 Features

- **Text Preprocessing**: Cleaning and preparing text data
- **TF-IDF Vectorization**: Converting text to numerical features
- **PassiveAggressiveClassifier**: Fast online learning algorithm
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Confusion Matrix**: Visual representation of model performance
- **Feature Importance**: Analysis of most important words for classification

## 📁 Project Structure
FakeNewsDetector/
├── fake_news_detector.py # Main training script
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── traindata.csv # Dataset (not included in repo)

text

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/tbm666/FakeNewsDetector.git
cd FakeNewsDetector
Install dependencies:

bash
pip install -r requirements.txt
📈 Usage
Run the main script to train and evaluate the model:

bash
python fake_news_detector.py
The script will:

Load and preprocess the dataset

Train the TF-IDF vectorizer

Train the PassiveAggressiveClassifier

Evaluate model performance

Generate visualizations

🔧 Model Details
Data Preprocessing
Text cleaning and normalization

TF-IDF vectorization with English stop words

Train-test split with stratification

Algorithm
PassiveAggressiveClassifier: Online learning algorithm that remains passive for correct classifications and turns aggressive for mistakes

TF-IDF: Term Frequency-Inverse Document Frequency for text representation

Performance
Accuracy: > 90%

Detailed classification report

Confusion matrix visualization

Feature importance analysis

📊 Results
The model achieves high accuracy in distinguishing between real and fake news articles. Key performance metrics include:

High precision and recall for both classes

Clear separation in feature importance

Robust performance on test data

🤝 Contributing
Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
tbm666

GitHub: @tbm666

🙏 Acknowledgments
Dataset sources and contributors

Scikit-learn library for machine learning tools

Open source community for continuous support