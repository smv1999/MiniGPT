from tensorflow.keras import mixed_precision
import tensorflow as tf
from models.models import TransformerBlock
from utils.utility import causal_attention_mask
import keras

# ---------- Mixed precision ----------
mixed_precision.set_global_policy("mixed_float16")
print("Mixed precision enabled:", mixed_precision.global_policy())


@keras.saving.register_keras_serializable(package="CustomModels")
class TinyGPT(tf.keras.Model):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers, mlp_ratio, dropout, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.token_emb = tf.keras.layers.Embedding(
            vocab_size, embed_dim, dtype="float32")
        self.pos_emb = tf.Variable(
            tf.zeros([1, seq_len, embed_dim], dtype=tf.float32), trainable=True)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.blocks = [TransformerBlock(
            embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, dtype="float32")
        self.head = tf.keras.layers.Dense(vocab_size, dtype="float32")

    def call(self, idx, training=None):
        B, T = tf.shape(idx)[0], tf.shape(idx)[1]
        tok = self.token_emb(idx)
        x = tok + self.pos_emb[:, :T, :]
        x = self.drop(x, training=training)
        mask = causal_attention_mask(B, T)
        for block in self.blocks:
            x = block(x, training=training, mask=mask)
        x = self.ln_f(x)
        return self.head(x)
