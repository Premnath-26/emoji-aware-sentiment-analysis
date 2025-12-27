# Pre-trained Embeddings

This project uses external pre-trained embeddings.
They are not included in the repository because they are very large.

## Step 1: Download GloVe embeddings

Run the following commands:

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

Use this file:
glove.6B.300d.txt

## Step 2: Download Emoji2Vec embeddings

Run this command:

wget https://github.com/uclnlp/emoji2vec/raw/master/pre-trained/emoji2vec.txt

## After download

Your folder should contain:

embeddings/
- glove.6B.300d.txt
- emoji2vec.txt
