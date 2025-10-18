import os
import tensorflow as tf
import sentencepiece as spm
from mini_gpt import MiniGPT
from utils.constants import *
from utils.utility import create_dataset, sample_sequence

# ---------- Load SentencePiece tokenizer ----------
SP_MODEL = "spm.model"
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL)
VOCAB_SIZE = sp.get_piece_size()
print(f"✅ Loaded SentencePiece model with vocab size: {VOCAB_SIZE}")

CHECKPOINT_DIR = "./checkpoints"


def train_model():
    dataset = create_dataset(sp, "gpt_model_datasets/final_dataset/corpus.txt")

    model = MiniGPT(VOCAB_SIZE, SEQ_LEN, EMBED_DIM,
                    NUM_HEADS, NUM_LAYERS, MLP_RATIO, DROPOUT)
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(EPOCHS):
        for step, (x, y) in enumerate(dataset):
            print("$%$%$", step)
            loss = train_step(x, y)
            if step % 10 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.numpy():.4f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.save_weights(os.path.join(
        CHECKPOINT_DIR, "minigpt_full_model.weights.h5"))
    print("✅ Training completed and model saved.")

    # Sampling
    seed_text = "An algorithm is"
    seed_ids = sp.encode(seed_text, out_type=int)
    generated_ids = sample_sequence(model, seed_ids, 50)
    print("🪄 Generated:", sp.decode(generated_ids))


if __name__ == "__main__":
    train_model()
