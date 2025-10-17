import numpy as np
import tensorflow as tf
from utils.constants import SEQ_LEN, BATCH_SIZE

AUTOTUNE = tf.data.AUTOTUNE

# ---------- Utility: causal mask ----------


def causal_attention_mask(batch_size, seq_len):
    i = tf.range(seq_len)[:, None]
    j = tf.range(seq_len)
    mask = tf.cast(i >= j, tf.int32)
    mask = tf.reshape(mask, (1, 1, seq_len, seq_len))
    return tf.tile(mask, [batch_size, 1, 1, 1])


# ---------- Sampling ----------


def top_k_logits(logits, k):
    values, _ = tf.math.top_k(logits, k=k)
    min_values = values[:, -1, tf.newaxis]
    return tf.where(logits < min_values, -1e10, logits)


def sample_sequence(model, seed_tokens, length, temperature=1.0, top_k=50):
    cur = np.array(seed_tokens, dtype=np.int32)[None, :]
    for _ in range(length):
        if cur.shape[1] > model.seq_len:
            cur = cur[:, -model.seq_len:]
        logits = model(cur, training=False)
        last_logits = logits[0, -1, :] / temperature
        last_logits = top_k_logits(last_logits[tf.newaxis, :], top_k)[0]
        probs = tf.nn.softmax(last_logits)
        next_id = tf.random.categorical(tf.math.log(
            probs)[tf.newaxis, :], 1)[0, 0].numpy()
        cur = np.concatenate([cur, [[next_id]]], axis=1)
    return cur[0].tolist()

# ---------- Create dataset ----------


def create_dataset(sp, file_path, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = sp.encode(text, out_type=int)
    print(f"Encoded {len(ids)} tokens from corpus.")

    n_windows = (len(ids) - 1) // seq_len
    inputs = []
    targets = []
    for i in range(n_windows):
        start = i * seq_len
        inputs.append(ids[start:start + seq_len])
        targets.append(ids[start + 1:start + 1 + seq_len])

    inputs = np.array(inputs, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    ds = ds.shuffle(1000).batch(
        batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds
