# emoji-aware-sentiment-analysis
Emoji-aware Sentiment Analysis using BiLSTM-CNN with GloVe & Emoji2Vec

This project implements an emoji-aware sentiment analysis system by combining
textual features and emoji semantics using deep learning.

## 🔍 Key Features
- BiLSTM + CNN architecture
- GloVe word embeddings
- Emoji2Vec embeddings
- Explicit emoji sentiment scoring
- LIME explainability
- Product brand sentiment ranking

## 📊 Datasets Used
- Sentiment140 (1.6M tweets)
- Emoji-based tweet dataset

> Datasets are not included due to size constraints.

## 🧠 Model Architecture
- Text → Embedding → BiLSTM → CNN → Pooling
- Emoji sentiment score concatenated before classification

## Model Performance

After training the emoji-aware sentiment analysis model on a combined dataset of Sentiment140 and custom emoji tweets, the evaluation results on the validation set are:

Classification Report:
| Class            | Precision | Recall | F1-Score | Support   |
| ---------------- | --------- | ------ | -------- | --------- |
| 0                | 0.79      | 0.83   | 0.81     | 9602      |
| 1                | 0.82      | 0.78   | 0.80     | 9601      |
| **Accuracy**     | —         | —      | **0.80** | **19203** |
| **Macro Avg**    | 0.80      | 0.80   | 0.80     | 19203     |
| **Weighted Avg** | 0.80      | 0.80   | 0.80     | 19203     |

Confusion Matrix:
| Actual \ Predicted | 0    | 1    |
| ------------------ | ---- | ---- |
| **0**              | 7934 | 1668 |
| **1**              | 2089 | 7512 |

Overall Metrics:

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.8044 |
| Precision | 0.8183 |
| Recall    | 0.7824 |
| F1 Score  | 0.8000 |

Interpretation:
The model achieves around 80% accuracy, with a balanced performance for both positive and negative classes. It can effectively predict sentiment while accounting for emoji information in tweets.

## ⚙️ Setup Instructions

```bash
pip install -r requirements.txt
