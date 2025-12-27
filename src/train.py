from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from src.embeddings import build_embedding_matrix
from src.model import build_model

def train_pipeline(df, glove_path, emoji2vec_path):
    MAX_LEN = 100
    EMBED_DIM = 300
    VOCAB_SIZE = 20000

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['clean_text'])

    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    padded = pad_sequences(sequences, maxlen=MAX_LEN)

    emoji_scores = df['emoji_score'].values.reshape(-1, 1)
    labels = to_categorical(df['label'], 2)

    Xtr, Xv, Etr, Ev, ytr, yv = train_test_split(
        padded, emoji_scores, labels, test_size=0.2, random_state=42
    )

    embedding_matrix = build_embedding_matrix(
        tokenizer, EMBED_DIM, glove_path, emoji2vec_path, VOCAB_SIZE
    )

    model = build_model(MAX_LEN, VOCAB_SIZE, EMBED_DIM, embedding_matrix)

    model.fit(
        {"input_1": Xtr, "input_2": Etr},
        ytr,
        validation_data=({"input_1": Xv, "input_2": Ev}, yv),
        epochs=10,
        batch_size=64,
        callbacks=[EarlyStopping(patience=3)]
    )

    return model, tokenizer, (Xv, Ev, yv)
