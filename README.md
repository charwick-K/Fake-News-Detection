# 📰 Fake News Detection Using Deep Learning

## 🔍 Introduction

In an era where misinformation spreads rapidly across digital platforms, the ability to detect fake news has become more critical than ever. This project presents a deep learning-based solution for fake news detection, leveraging a rich dataset from Kaggle and evaluating multiple neural network architectures. The goal is to build reliable models that can accurately classify news articles as real or fake, thereby contributing to the fight against online misinformation.

## 📂 Dataset Description

The dataset used in this project is sourced from Kaggle and is composed of two primary CSV files:

- **`real.csv`**: Contains 21,417 legitimate news articles.
- **`fake.csv`**: Contains 23,481 fabricated or misleading news articles.

Each entry includes the article's title, text, and other metadata, providing a solid foundation for training and evaluating classification models.

## 🧠 Deep Learning Models Explored

To assess the effectiveness of various neural network architectures, the following models were implemented and compared:

- **LSTM (Long Short-Term Memory)**: Captures long-range dependencies in text sequences.
- **Bidirectional LSTM**: Enhances LSTM by processing input in both forward and backward directions.
- **GRU (Gated Recurrent Unit)**: A lighter alternative to LSTM with comparable performance.
- **CNN (Convolutional Neural Network)**: Extracts local patterns in text using convolutional filters.
- **ResNet (Residual Network)**: Adapted for text classification to explore deep residual learning.
- **COBRA (Contextualized Bidirectional Recurrent Architecture)**: A hybrid model designed for contextual understanding of text.

## 📊 Model Performance

The models were evaluated using standard metrics such as accuracy, precision, recall, and F1-score. Key findings include:

| Model               | Accuracy |
|--------------------|----------|
| LSTM               | 0.97     |
| Bidirectional LSTM | 0.97     |
| GRU                | 0.97     |
| COBRA              | 0.97     |
| CNN                | 0.89     |
| ResNet             | 0.76     |

These results highlight the superior performance of recurrent architectures, particularly LSTM-based models, in capturing the nuances of textual data for fake news detection.

## 🛠️ Getting Started

### 🔧 Prerequisites

Ensure the following dependencies are installed:

- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn

### 📥 Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
```

### 🚀 Running the Models

Use the provided Jupyter notebooks or Python scripts to train and evaluate the models. Each notebook includes step-by-step instructions for:

- Data preprocessing and cleaning
- Tokenization and embedding
- Model training and validation
- Performance evaluation and visualization

## 📈 Results Visualization

After training, the models output detailed performance metrics including:

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curves
- Word Clouds for most frequent terms in real vs. fake news

These visualizations help interpret model behavior and identify areas for improvement.

## 🧩 Contributions & Insights

This project not only benchmarks multiple deep learning models but also sheds light on their relative strengths and weaknesses in the context of fake news detection. Key insights include:

- Recurrent models (LSTM, GRU, BiLSTM) outperform CNN and ResNet for sequential text data.
- COBRA’s contextual awareness enhances classification accuracy.
- Proper preprocessing (e.g., stopword removal, stemming) significantly boosts model performance.

These findings can inform future research and development of AI-driven misinformation detection systems.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## 🤝 Contributing

Contributions are welcome! If you’d like to improve this project, feel free to:

- Fork the repository
- Submit issues or feature requests
- Open a pull request with enhancements

## 📬 Contact

For questions, suggestions, or collaboration inquiries, reach out to:

📧 **charwick14@gmail.com**
