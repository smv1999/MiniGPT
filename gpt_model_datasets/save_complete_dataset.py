import glob

files = ["datasets/english_greetings.txt",
         "datasets/computer_science.txt", "datasets/cs_qa_snippets.txt"]

with open("final_dataset/corpus.txt", "w", encoding="utf-8") as out:
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read().strip()
            # Basic cleanup
            text = text.replace('\n\n', '\n').replace('\r', '')
            out.write(text + "\n\n")
print("✅ Final corpus saved to data/corpus.txt")
