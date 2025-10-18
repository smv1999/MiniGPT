from traceback import print_tb
import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent='MiniGPT-CS/1.0 (ComputerScienceLLM; research; +https://huggingface.co)'
)

with open("datasets/cs_topics.txt", 'r') as f:
    topics = f.read().splitlines()

with open("datasets/computer_science.txt", "w", encoding="utf-8") as f:
    for topic in topics:
        page = wiki.page(topic)
        if page.exists():
            f.write(page.text + "\n\n")
print("✅ Saved computer_science.txt")
