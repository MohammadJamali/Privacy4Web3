import os
import shutil
import sys
import json
import gzip
import pickle
import zipfile
from shutil import rmtree

import tensorflow as tf
from tensorflow.keras import layers


def generate(prompt: str, model, tokenizer, output_len: int):
    print(f"Generating response based on prompt: {prompt}")

    input_tokens = tokenizer.texts_to_sequences([prompt])[0]
    sequence = []

    for _ in range(output_len):
        with tf.device('/cpu:0'):
            predictions = model(tf.convert_to_tensor([input_tokens]), training=False)[-1]
            predicted_token_id = tf.argmax(predictions, axis=-1).numpy()
            predicted_token = tokenizer.sequences_to_texts([[predicted_token_id]])
            sequence.append(predicted_token[0])

            input_tokens.append(predicted_token_id)
            del input_tokens[0]

    return ' '.join(sequence)


def normalize(corpus):
    import re
    corpus = re.sub(r'\n\r', ' ', corpus)
    corpus = re.sub(r'([,.!?;:])', r' \1 ', corpus)
    corpus = re.sub(r'\s+', ' ', corpus)
    corpus = corpus.strip().lower()

    return corpus


class PositionEncoding(layers.Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        return self.pos_embedding(positions)


class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, layer_index,
                 value_dim=None, dropout=0.1, use_bias=True,
                 seed=None, **kwargs):
        super().__init__(**kwargs)

        self.seed = seed
        self.dropout = dropout
        self.key_dim = key_dim
        self.use_bias = use_bias
        self.num_heads = num_heads
        self.layer_index = layer_index
        self.value_dim = value_dim if value_dim else key_dim

        self.key_kernel, self.query_kernel, self.value_kernel, self.output_kernel = (None,) * 4
        self.query_bias, self.key_bias, self.value_bias, self.output_bias = (None,) * 4

        self.dropout_layer = layers.Dropout(dropout)
        self.softmax = layers.Softmax(axis=-1)

        units = self.num_heads * self.key_dim
        name = f'atten_{self.layer_index}_%s_kernel'
        value_units = self.num_heads * self.value_dim
        initializer = tf.keras.initializers.GlorotUniform(seed=self.seed)

        args = (tf.float32, initializer)
        self.key_kernel = self.add_weight(name % 'key', (self.key_dim, units), *args)
        self.query_kernel = self.add_weight(name % 'query', (self.key_dim, units), *args)
        self.value_kernel = self.add_weight(name % 'value', (self.value_dim, value_units), *args)
        self.output_kernel = self.add_weight(name % 'output', (value_units, self.key_dim), *args)

        if self.use_bias:
            name = f'atten_{self.layer_index}_%s_bias'
            self.key_bias = self.add_weight(name % 'key', (units,), *args)
            self.query_bias = self.add_weight(name % 'query', (units,), *args)
            self.value_bias = self.add_weight(name % 'value', (value_units,), *args)
            self.output_bias = self.add_weight(name % 'output', (self.key_dim,), *args)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, key_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, key_dim)

    def linear_projection(self, x, kernel, bias=None):
        """Performs linear projection using the weight matrix."""
        x = tf.matmul(x, kernel)
        if bias is not None:
            x = x + bias
        return x

    def call(self, inputs, attention_mask=None, return_attention_scores=False, training=False, **kwargs):
        if len(inputs) == 3:
            query, value, key = inputs
        elif len(inputs) == 2:
            query, value = inputs
            key = None
        else:
            raise ValueError("inputs should have shape (batch_size, seq_len, num_heads[, key_dim])")

        batch_size = tf.shape(query)[0]

        # Linear projections
        # (batch_size, seq_len, num_heads * key_dim)
        query = self.linear_projection(query, self.query_kernel, self.query_bias)
        # (batch_size, seq_len, num_heads * key_dim)
        key = self.linear_projection(key or value, self.key_kernel, self.key_bias)
        # (batch_size, seq_len, num_heads * value_dim)
        value = self.linear_projection(value, self.value_kernel, self.value_bias)

        # Split heads
        key = self.split_heads(key, batch_size)  # (batch_size, num_heads, seq_len_k, key_dim)
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len_q, key_dim)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len_v, value_dim)

        # Scaled dot-product attention
        attention_scores = tf.matmul(query, key, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))

        if attention_mask is not None:
            attention_scores += (attention_mask * -1e9)  # Apply mask

        attention_weights = self.softmax(attention_scores)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_weights = self.dropout_layer(attention_weights, training=training)

        # Weighted sum of values
        attention_output = tf.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len_q, value_dim)

        # Combine heads
        # (batch_size, seq_len_q, num_heads, value_dim)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads * value_dim)
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.num_heads * self.value_dim))

        # Final linear layer
        # (batch_size, seq_len_q, key_dim)
        output = self.linear_projection(concat_attention, self.output_kernel, self.output_bias)

        if return_attention_scores:
            return output, attention_weights
        return output


class FeedForward(layers.Layer):
    def __init__(self, expansion: int, projection: int, id: str, seed=None, **kwargs):
        super().__init__(**kwargs)

        name = f'feedforward_{id}_%s_weights'
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)

        self.expansion_weights = self.add_weight(name % 'expansion', (projection, expansion), tf.float32, initializer)
        self.projection_weights = self.add_weight(name % 'projection', (expansion, projection), tf.float32, initializer)

        name = f'feedforward_{id}_%s_bias'
        initializer = tf.keras.initializers.Zeros()
        self.expansion_bias = self.add_weight(name % 'expansion', (expansion,), tf.float32, initializer)
        self.projection_bias = self.add_weight(name % 'projection', (projection,), tf.float32, initializer)

    def call(self, inputs):
        x = tf.nn.relu(tf.matmul(inputs, self.expansion_weights) + self.expansion_bias)
        output = tf.matmul(x, self.projection_weights) + self.projection_bias
        return output


class DecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, layer_index,
                 value_dim=None, dropout=0.1, use_bias=True, seed=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.attention = MultiHeadAttention(num_heads, embed_dim, layer_index, value_dim, dropout, use_bias, seed=seed)
        self.feedforward = FeedForward(ff_dim, embed_dim, layer_index, seed=seed)

        self.attn_layernorm = layers.LayerNormalization()
        self.feedforward_layernorm = layers.LayerNormalization()

        self.attn_dropout = layers.Dropout(dropout, seed=seed)
        self.feedforward_dropout = layers.Dropout(dropout, seed=seed)

    def call(self, inputs, training=False, mask=None, **kwargs):
        # Multi-Head Attention block
        attn_output = self.attention((inputs, inputs), training=training, attention_mask=mask)
        attn_output = self.attn_dropout(attn_output, training=training)
        attention = self.attn_layernorm(inputs + attn_output)

        # Feed-forward block
        feedforward_output = self.feedforward(attention)
        feedforward_output = self.feedforward_dropout(feedforward_output, training=training)
        output = self.feedforward_layernorm(attention + feedforward_output)

        return output


class TransformerDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim,
                 max_len, num_layers, value_dim=None, dropout=0.1,
                 use_bias=True, seed=None, **kwargs):
        super().__init__()

        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionEncoding(max_len, embed_dim)

        self.decoder_layers = [
            DecoderBlock(embed_dim, num_heads, ff_dim, layer_index, value_dim, dropout, use_bias, seed=seed, **kwargs)
            for layer_index in range(num_layers)
        ]

        # Final projection layer initialized with weights and bias
        name = 'transformer_feedforward_final_layer_%s'
        bias_initializer = tf.keras.initializers.Zeros()
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)

        self.projection_weights = self.add_weight(name % 'weights', (embed_dim, vocab_size), tf.float32, initializer)
        self.projection_bias = self.add_weight(name % 'bias', (vocab_size,), tf.float32, bias_initializer)

    def call(self, inputs, training=True, mask=None):
        # Compute embedding + positional encoding
        x = self.embedding(inputs) + self.pos_encoding(inputs)

        # Pass through the decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, training=training, mask=mask)

        # Final linear projection using manual weights and bias
        logits = tf.matmul(x, self.projection_weights) + self.projection_bias

        # Return the last token's logits
        return logits[:, -1, :]


def read_prompt(local=False):
    if local:
        prompt = 'the cat was a mystery'
        print(f"Setting prompt to : {prompt}")
        return prompt

    input_file_path = '/data/inputs/algoCustomData.json'
    if not os.path.exists(input_file_path):
        print("Provided input file does not exist.")
        return None

    with open(input_file_path, "r") as f:
        input_data = json.load(f)
        print(input_data)

    if 'prompt' not in input_data:
        print("Provided input file does not contain prompt.")
        return None

    prompt = input_data['prompt']
    if not prompt or len(prompt) < MAX_LEN:
        print('Prompt is too short')
        return None

    return prompt


def load_model(local=False):
    MODEL_PATH = 'tmp/model.weights.h5'
    CONFIG_PATH = 'tmp/model.json'
    TOKENIZER_PATH = 'tmp/model.tokenizer'

    if os.path.exists('tmp'):
        shutil.rmtree('tmp')
    os.mkdir('tmp')

    archive_path, plugin_path = None, None
    if local:
        print("Using local `cat_story` base model and `garfield` plugin-in")
        plugin_path = 'plugin.lora.garfield'
        archive_path = 'model.cat_story.zip'
    else:
        dids = os.getenv("DIDS", None)
        if not dids:
            print("No DIDs found in environment.")
            exit(1)

        dids = json.loads(dids)
        print('DIDS', dids)

        for did in dids:
            filename = "/data/ddos/" + did

            with open(filename) as json_file:
                ddo = json.load(json_file)
                print('DDO:\n', ddo)

            if ddo['metadata']['role'] == 'plugin':
                plugin_path = f"/data/inputs/{did}/0"
            elif ddo['metadata']['role'] == 'base':
                archive_path = f"/data/inputs/{did}/0"

    if not archive_path:
        print('Base weight must be included')
        exit(1)

    print(f"Extracting model...")
    with zipfile.ZipFile(archive_path, 'r') as zip_file_handle:
        zip_file_handle.extractall('tmp/')

    print("Loading tokenizer...")
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("Loading model config...")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    model = tf.keras.saving.deserialize_keras_object(config, {'TransformerDecoder': TransformerDecoder})

    print("Loading model...")
    model.load_weights(MODEL_PATH)

    plugin = None
    if plugin_path:
        print("Loading LoRA weights...")

        with gzip.open(plugin_path, 'rb') as file_handle:
            file_handle.seek(0)
            plugin = pickle.load(file_handle)

    shutil.rmtree('tmp')

    return model, tokenizer, plugin


def apply_weights(model, weights):
    import numpy as np

    print("Applying plug-in...")
    for weight_name in weights:
        for weight in model.weights:
            if weight.name == weight_name:
                reconstructed = np.dot(weights[weight_name]['A'], weights[weight_name]['B'])
                weight.assign_add(reconstructed)
                break

    return model


def write_output(output: str):
    print("Writing output...")
    filename = "output.txt" if local else "/data/outputs/result"
    with open(filename, "w") as f:
        f.write(output)
    print(output)


def run_algo(local):
    prompt = read_prompt(local)
    if not prompt:
        return

    model, tokenizer, plugin = load_model(local)
    if plugin:
        model = apply_weights(model, plugin)

    output = generate(prompt, model, tokenizer, 20)
    write_output(output)


def get_filepaths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_paths.append(os.path.join(root, filename))

    return list(file_paths)


if __name__ == '__main__':
    MAX_LEN = 5
    print("/data\n", '\n'.join(get_filepaths('/data')))

    local = len(sys.argv) == 2 and sys.argv[1] == "local"

    run_algo(local)
