# src/model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, Concatenate
)

def build_model(max_len, vocab_size, embed_dim, embedding_matrix):
    text_input = Input(shape=(max_len,))
    emoji_input = Input(shape=(1,))

    embed = Embedding(
        vocab_size, embed_dim,
        weights=[embedding_matrix],
        trainable=False
    )(text_input)

    bilstm = Bidirectional(LSTM(128, return_sequences=True))(embed)
    conv = Conv1D(64, 3, activation='relu')(bilstm)
    pool = GlobalMaxPooling1D()(conv)

    concat = Concatenate()([pool, emoji_input])
    drop = Dropout(0.3)(concat)
    dense = Dense(64, activation='relu')(drop)
    output = Dense(2, activation='softmax')(dense)

    model = Model([text_input, emoji_input], output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
