import os
import re

RAW_DIR = "data/raw/wiki"
CLEAN_DIR = "data/clean/wiki"
os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_text(text):
    # Remove bracketed references like [1], [12], etc.
    text = re.sub(r"\[\d+\]", "", text)

    # Remove headings and TOC (=== Header ===)
    text = re.sub(r"={2,}.*?={2,}", "", text)

    # Remove multiple spaces and normalize newlines
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Remove empty lines and short lines
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
    return "\n".join(lines)

def process_all():
    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".txt"):
            continue

        raw_path = os.path.join(RAW_DIR, filename)
        clean_path = os.path.join(CLEAN_DIR, filename)

        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                text = f.read()

            cleaned = clean_text(text)

            with open(clean_path, "w", encoding="utf-8") as f:
                f.write(cleaned)

            print(f"[+] Cleaned: {filename}")
        except Exception as e:
            print(f"[!] Error processing {filename}: {e}")

if __name__ == "__main__":
    process_all()
