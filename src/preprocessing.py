import re
import emoji
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor', 'but'}
tokenizer_nltk = TweetTokenizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s" + re.escape(''.join(emoji.EMOJI_DATA.keys())) + "]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = tokenizer_nltk.tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def compute_emoji_score(text, emoji_score_map=None):
    emojis = [c for c in text if emoji.is_emoji(c)]
    if not emojis or emoji_score_map is None:
        return 0.0
    return np.mean([emoji_score_map.get(e, 0) for e in emojis])

def preprocess_dataset(df, text_col, label_col):
    df['clean_text'] = df[text_col].apply(clean_text)
    df['emoji_score'] = df[text_col].apply(lambda x: compute_emoji_score(x))
    df['label'] = df[label_col].apply(lambda x: 1 if x in [1, 3, 4] else 0)
    return df[['clean_text', 'emoji_score', 'label']]
