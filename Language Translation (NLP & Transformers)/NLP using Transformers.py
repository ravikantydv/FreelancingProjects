# ## Part 3: NLP using Transformers 
# 
# In Part 3 we will look at the Transformer architecture and how it can be used in a specific NLP task, machine translation (from English to Spanish).
# 
# Please refer to Chapter 11 of [our textbook](https://learning.oreilly.com/library/view/deep-learning-with/9781617296864) for additional information.
# 

# Useful sources and references for Part 3:
# 
# - https://colab.research.google.com/github/fchollet/deep-learning-with-python-notebooks/blob/master/chapter11_part03_transformer.ipynb
# - https://colab.research.google.com/github/fchollet/deep-learning-with-python-notebooks/blob/master/chapter11_part04_sequence-to-sequence-learning.ipynb 
# 

# ### Setup

# In[2]:


get_ipython().system('wget http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip')
get_ipython().system('unzip -q spa-eng.zip')


# In[3]:


text_file = "spa-eng/spa.txt"
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    english, spanish = line.split("\t")
    spanish = "[start] " + spanish + " [end]"
    text_pairs.append((english, spanish))


# In[5]:


import random
print(random.choice(text_pairs))


# In[6]:


import random
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]


# **Vectorizing the English and Spanish text pairs**

# In[7]:


import tensorflow as tf
import string
import re

from tensorflow import keras
from tensorflow.keras import layers

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

vocab_size = 15000
sequence_length = 20

source_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_spanish_texts)


# **Preparing datasets for the translation task**

# In[8]:


batch_size = 64

def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return ({
        "english": eng,
        "spanish": spa[:, :-1],
    }, spa[:, 1:])

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)


# In[9]:


for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"inputs['spanish'].shape: {inputs['spanish'].shape}")
    print(f"targets.shape: {targets.shape}")


# ### The Transformer encoder

# In[10]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config


# ### The Transformer decoder

# In[11]:


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)


# ### Putting it all together: A Transformer for machine translation

# **PositionalEmbedding layer**

# In[12]:


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config


# **End-to-end Transformer**

# In[13]:


embed_dim = 256
dense_dim = 2048
num_heads = 8

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="spanish")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)


# **Training the sequence-to-sequence Transformer**

# In[13]:


transformer.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
transformer.fit(train_ds, epochs=10, validation_data=val_ds)


# In[17]:


# Save the compiled model
transformer.save('compiled_model.h5')


# **Translating new sentences with our Transformer model**

# In[22]:


import numpy as np
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
            [decoded_sentence])[:, :-1]
        predictions = transformer(
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(20):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))


# ## Conclusions
# 
# In this code, we have implemented a transformer model for sequence-to-sequence translation, which is a popular application of deep learning in natural language processing. The model is trained on a parallel corpus of English and Spanish sentences and is able to translate English sentences to Spanish.
# 
# We have used the TensorFlow framework to build the model, which consists of an encoder and a decoder. The encoder consists of multiple TransformerEncoder layers, each of which applies multi-head attention and a feed-forward network to the input. The decoder consists of multiple TransformerDecoder layers, each of which applies multi-head attention to the encoded input and a feed-forward network to the decoder input. We have also used masking to ensure that the model attends only to the relevant parts of the input.
# 
# We have preprocessed the input data using the TextVectorization layer, which converts text to integer sequences, and split the data into training, validation, and test sets. We have also implemented a custom standardization function to remove punctuation from the Spanish text.
# 
# The model was trained on the training set and evaluated on the validation set. The performance of the model was measured using the categorical cross-entropy loss and the accuracy. The model achieved good performance on both the training and validation sets.
# 
# Overall, this code provides a good example of how to implement a transformer model for sequence-to-sequence translation in TensorFlow. It demonstrates the importance of preprocessing and how to use masking to handle variable-length input sequences.

# ### 

# ### Lets play with it and tweak it in ways that make more sense.
# 
# 
# Here are some ways you can tweak the code:
# 
# 1. Change the size of the vocabulary: 
# The code uses a vocabulary size of 15,000 words for both source and target languages. You can increase or decrease this number to experiment with the model's performance.
# 
# 2. Change the number of layers and the number of heads in the Transformer: 
# The code uses a single layer with 8 heads for both the encoder and the decoder. You can increase or decrease the number of layers and the number of heads to experiment with the model's performance.
# 
# 3. Change the size of the embedding and dense layers: 
# The code uses an embedding dimension of 256 and a dense layer dimension of 512. You can increase or decrease these numbers to experiment with the model's performance.
# 
# 4. Use different pre-processing methods: 
# The code applies custom standardization to the input text using a regular expression to remove punctuation and convert all characters to lowercase. You can experiment with different pre-processing techniques, such as stemming, lemmatization, or stop word removal.
# 
# 5. Change the dataset: 
# You can download and use a different dataset for the translation task. You can also use the same dataset to train a model for a different NLP task, such as text classification, sentiment analysis, or question answering.

# In[24]:


get_ipython().system('wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en')
get_ipython().system('wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi')


# In[39]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.losses import sparse_categorical_crossentropy
# Function to get the transformer encoder
def get_transformer_encoder(num_layers, input_vocab_size, embed_dim, dense_dim, num_heads, dropout_rate=0.1):
    inputs = Input(shape=(None,))
    embedding_layer = Embedding(input_vocab_size, embed_dim)(inputs)
    positional_encoding_layer = get_positional_encoding(input_vocab_size, embed_dim)(inputs)
    x = embedding_layer + positional_encoding_layer
    for i in range(num_layers):
        x = get_transformer_encoder_layer(embed_dim, dense_dim, num_heads, dropout_rate)(x)
    return Model(inputs, x)

# Function to get a single transformer encoder layer
def get_transformer_encoder_layer(embed_dim, dense_dim, num_heads, dropout_rate=0.1):
    inputs = Input(shape=(None, embed_dim))
    attention_output = get_attention_layer(num_heads)(inputs)
    x1 = tf.keras.layers.Add()([attention_output, inputs])
    x2 = get_layer_normalization()(x1)
    x3 = get_feedforward_layer(dense_dim)(x2)
    x4 = tf.keras.layers.Dropout(dropout_rate)(x3)
    x5 = tf.keras.layers.Add()([x4, x2])
    return Model(inputs, x5)

# Function to get the attention layer
def get_attention_layer(num_heads):
    inputs = Input(shape=(None, None))
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_heads)(inputs, inputs)
    return Model(inputs, attention)

# Function to get the layer normalization layer
def get_layer_normalization():
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)

# Function to get the feedforward layer
def get_feedforward_layer(dense_dim):
    return tf.keras.Sequential([
        Dense(dense_dim, activation='relu'),
        Dense(dense_dim)
    ])

# Function to get the positional encoding
def get_positional_encoding(input_vocab_size, embed_dim):
    positions = np.arange(input_vocab_size)[:, np.newaxis]
    embed_positions = np.arange(embed_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (embed_positions // 2)) / np.float32(embed_dim))
    angle_rads = positions * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    positional_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(positional_encoding, dtype=tf.float32)

# Function to preprocess the input and output data
def preprocess_data(input_texts, target_texts, input_vocab_size, target_vocab_size):
    # Create input and target tokenizers
    input_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=input_vocab_size, oov_token='<OOV>')
    input_tokenizer.fit_on_texts(input_texts)
    target_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=target_vocab_size, oov_token='<OOV>')
    target_tokenizer.fit_on_texts(target_texts)

    # Convert input and target texts to sequences of integers
    input_sequences = input_tokenizer.texts_to_sequences(input_texts)
    target_sequences = target_tokenizer.text


# In[33]:


def get_transformer_decoder(num_layers, output_vocab_size, embed_dim, dense_dim, num_heads):
    inputs = keras.Input(shape=(None,), dtype="int64")
    encoded_seq = keras.Input(shape=(None, embed_dim))
    attention_mask = keras.layers.Lambda(lambda x: tf.cast(tf.math.not_equal(x, 0), dtype=tf.float32))(inputs)
    
    padding_mask = get_padding_mask(inputs)
    look_ahead_mask = get_look_ahead_mask(inputs)
    decoder_mask = tf.maximum(padding_mask, look_ahead_mask)
    
    x = keras.layers.Embedding(output_vocab_size, embed_dim)(inputs)
    x = x * tf.math.sqrt(tf.cast(embed_dim, tf.float32))
    x = x + PositionalEncoding(output_vocab_size, embed_dim)(x)
    
    for i in range(num_layers):
        x = get_transformer_decoder_layer(embed_dim, dense_dim, num_heads, f"decoder_layer_{i + 1}")([x, encoded_seq, decoder_mask])
    
    x = keras.layers.Dense(output_vocab_size, name="decoder_output")(x)
    return keras.Model(inputs=[inputs, encoded_seq], outputs=x, name="transformer_decoder")


# In[14]:


import tensorflow as tf
import string
import re

from tensorflow import keras
from tensorflow.keras import layers

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

vocab_size = 30000
sequence_length = 40

source_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_spanish_texts)


# In[15]:


embed_dim = 512
dense_dim = 2048
num_heads = 10

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="spanish")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)


# Fitting Transformer on different Optimizer after changing the size of vocab, training set, embeddings etc..

# In[16]:


transformer.compile(
    optimizer="sgd",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
transformer.fit(train_ds, epochs=10, validation_data=val_ds)


# In[17]:


import numpy as np
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 40

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
            [decoded_sentence])[:, :-1]
        predictions = transformer(
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(20):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))


# Using a different Optimizer and checking results--

# In[18]:


transformer.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
transformer.fit(train_ds, epochs=10, validation_data=val_ds)


# In[19]:


import numpy as np
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 40

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
            [decoded_sentence])[:, :-1]
        predictions = transformer(
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(20):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))


# We saw that the results are better when we use rmsprop optimizer in the comparision of sgd and adam.
# With rmsprop, we saw greater accuracy also.

# In[20]:


#### DONE
