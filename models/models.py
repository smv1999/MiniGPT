import tensorflow as tf


class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.proj = tf.keras.layers.Dense(embed_dim)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=None, mask=None):
        attn_mask = tf.cast(tf.squeeze(mask, axis=1),
                            tf.bool) if mask is not None else None
        y = self.mha(query=x, value=x, key=x,
                     attention_mask=attn_mask, training=training)
        y = self.dropout(self.proj(y), training=training)
        return y


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.ln1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, dtype="float32")
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, dtype="float32")
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(
                embed_dim * mlp_ratio, activation=tf.keras.activations.gelu, dtype="float32"),
            tf.keras.layers.Dense(embed_dim, dtype="float32"),
            tf.keras.layers.Dropout(dropout)
        ])

    def call(self, x, training=None, mask=None):
        x = x + self.attn(self.ln1(x), training=training, mask=mask)
        x = x + self.mlp(self.ln2(x), training=training)
        return x
