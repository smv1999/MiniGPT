import sentencepiece as spm

# Path to your text corpus (plain text)
input_file = "gpt_model_datasets/final_dataset/corpus.txt"

# Train a SentencePiece model
spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix="spm",
    vocab_size=8000,
    character_coverage=1.0,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

print("✅ SentencePiece model trained: spm.model, spm.vocab")
