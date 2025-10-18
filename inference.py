from keras.models import load_model
import sentencepiece as spm
from mini_gpt import MiniGPT
from utils.constants import *
import tensorflow as tf

SP_MODEL = "spm.model"
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL)

# 2️⃣ Recreate model and load weights
vocab_size = sp.get_piece_size()
model = MiniGPT(vocab_size, SEQ_LEN, EMBED_DIM,
                NUM_HEADS, NUM_LAYERS, MLP_RATIO, DROPOUT)
model.build((None, 128))
model.load_weights("checkpoints/minigpt_full_model.weights.h5")


# 3️⃣ Prepare input text (question/prompt)
prompt = "What is a compiler?"
input_ids = sp.encode(prompt, out_type=int)


# Generate up to 50 new tokens
max_new_tokens = 50
for _ in range(max_new_tokens):
    input_tensor = tf.constant([input_ids])
    logits = model(input_tensor)

    # Get predicted next token (greedy)
    next_token_id = int(tf.argmax(logits[:, -1, :], axis=-1).numpy()[0])

    # Stop if end-of-sentence token (optional)
    if next_token_id == sp.eos_id():
        break

    # Append and continue
    input_ids.append(next_token_id)

# Decode all tokens to text
output_text = sp.decode(input_ids)
print("\n🧠 Model response:\n", output_text)
