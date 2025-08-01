import os
import json
from tokenizer.mabpe import MARegexTokenizer
from datasets import Dataset
from typing import Optional

class MABPECorpusBuilder:
    def __init__(
        self,
        clean_text_dir: str,
        tokenizer_path: str,
        vocab_size: int = 5000
    ):
        self.clean_text_dir = clean_text_dir
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        special_tokens = {
            '<|endoftext|>': 100257,
            '<|fim_prefix|>': 100258,
            '<|fim_middle|>': 100259,
            '<|fim_suffix|>': 100260,
            '<|endofprompt|>': 100276
            }
        self.tokenizer = MARegexTokenizer(vocab_size=self.vocab_size
                                          ,special_tokens=special_tokens)


    def train_or_load_tokenizer(self, force_retrain=False):
        if os.path.exists(self.tokenizer_path) and not force_retrain:
            print(f"[+] Loading tokenizer from {self.tokenizer_path}")
            self.tokenizer.load(self.tokenizer_path)
        else:
            print("[+] Training BPE tokenizer from scratch...")
            self.tokenizer.build_bpe(self.clean_text_dir)
            self.tokenizer.save(self.tokenizer_path)
            print(f"[✓] Tokenizer saved to {self.tokenizer_path}")
