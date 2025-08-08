import os
import regex as re
from collections import defaultdict, Counter
from tqdm import tqdm
import json
import unicodedata
import concurrent.futures
import numpy as np
import time
import yaml

# Efficient thread-safe logging setup
import logging
logger = logging.getLogger("bpe_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
file_handler = logging.FileHandler("bpe_log.txt", mode="a", encoding="utf-8")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(threadName)s %(message)s')
handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
    logger.addHandler(file_handler)



class MATokenizer:
    
    def load_config(self):
        with open("config.yaml", "r") as f:
            GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
            BATCHSIZE = 100
            CHUNKING_BATCHSIZE = 10000
            CHUNKING_MAX_WORKER = 50
            MERGE_BATCHING_SIZE = 10000
            MERGE_BATCHING_MAX_WORKER = 50
            MAX_WORKER = 10
            config = yaml.safe_load(f)
            self.GPT2_SPLIT_PATTERN = config['train_tokenizer'].get("gpt2_split_pattern", GPT2_SPLIT_PATTERN)
            self.GPT4_SPLIT_PATTERN = config['train_tokenizer'].get("gpt4_split_pattern", GPT4_SPLIT_PATTERN)
            self.BATCHSIZE = config['train_tokenizer'].get("batch_size", BATCHSIZE)
            self.CHUNKING_BATCHSIZE = config['train_tokenizer'].get("chunking_batch_size", CHUNKING_BATCHSIZE)
            self.CHUNKING_MAX_WORKER = config['train_tokenizer'].get("chunking_max_worker", CHUNKING_MAX_WORKER)
            self.MERGE_BATCHING_SIZE = config['train_tokenizer'].get("merge_batching_size", MERGE_BATCHING_SIZE)
            self.MERGE_BATCHING_MAX_WORKER = config['train_tokenizer'].get("merge_batching_max_worker", MERGE_BATCHING_MAX_WORKER)
            self.MAX_WORKER = config['train_tokenizer'].get("max_worker", MAX_WORKER)
            return config

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.special_tokens = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab = self._build_vocab()
        self.pattern = ""
        config = self.load_config()

    def batching_files(self,folder_path,batchsize):
        all_files = os.listdir(folder_path)
        total = len(all_files)
        return [all_files[offset - batchsize : offset ] for offset in range(batchsize,total+batchsize,batchsize) if all_files[offset - batchsize : offset]]

    def read_corpus_batches(self,batch, folder_path):
        batch_text = ""
        i = 0
        for fname in batch:
            if fname.endswith(".txt"):
                with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                    batch_text += f.read().lower() + "\n"
                i+=1
                print(f'processed {i}    / {len(batch)} ...'  , end="\r")
        return batch_text

    def read_corpus(self, folder_path):
        text = ""
        print("Location of clean data" , folder_path)
        batches = self.batching_files(folder_path,self.BATCHSIZE)
        threads = []
        with concurrent.futures.ThreadPoolExecutor(self.MAX_WORKER) as executor:
            for batch in batches:    
                threads.append(executor.submit(self.read_corpus_batches,batch,folder_path))
            for idx,thread in enumerate( concurrent.futures.as_completed(threads)):
                text += thread.result() 
                print(f'processed thread {idx+1} / {len(batches)} ...')
        return text

    def get_pairs(self, ids , pairs=None):
        """
        iterate over all the pairs and calculate the frequency for all of them
        """
        pairs = defaultdict(int) if pairs is None else pairs
        for i in range(len(ids)-1):
                pair = (ids[i], ids[i+1])
                pairs[pair] += 1
        return pairs
    
    def get_pairs_counter(self, ids , pairs=None):
        """
        iterate over all the pairs and calculate the frequency for all of them
        """
        pairs = Counter() if pairs is None else pairs
        arr = ids
        local_pairs = Counter( list(zip(arr[:-1], arr[1:])))
        pairs+=local_pairs
        return pairs
    
    def get_pairs_parallel(self, ids):
        """
        iterate over all the pairs and calculate the frequency for all of them
        """
        arr = ids
        pairs = list(zip(arr[:-1], arr[1:]))
        return Counter(pairs)

    def merge_tokens(self,ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i+=2
            else:
                new_ids.append(ids[i])
                i +=1
        return new_ids
    
    def fast_merge_tokens(self,ids, pair, idx):
        mask = (ids[:-1] == pair[0]) & (ids[1:] == pair[1])
        # if no pair found, return original ids
        if np.sum(mask) == 0:
            return ids
        i = 0
        n = len(ids)
        new_ids = []
        while i < n:
            if i < n-1 and mask[i]:
                new_ids.append(idx)
                i+=2
            else:
                new_ids.append(ids[i])
                i +=1
        return np.array(new_ids)

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def replace_cotnrol_character(s):
        chars = []
        for ch in s:
            if unicodedata.category(ch)[0] != "c":
                chars.append(ch)
            else:
                chars.append(f"\\u{ord(ch):04x}") # excape
        return "".join(chars)

    def build_bpe(self, corpus_path):
        return NotImplementedError

    def encode(self, text):
        # words = re.findall(r"\w+|\S", text.lower())
        # return [self.encode_word(word) for word in words]
        return NotImplementedError

    def decode(self, ids):
        return NotImplemented

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocabsize": self.vocab_size,
                "pattern": self.pattern ,
                "merges": [[key[0] , key[1] , val] for key, val in self.merges.items()],
                "special_tokens": self.special_tokens
            }, f, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.merges = {(m[0],m[1]):m[2] for m in sorted(data["merges"], key=lambda x: x[2]) }
            self.special_tokens = data['special_tokens']
            self.inverse_special_tokens = {v:k for k,v in self.special_tokens.items()}
            self.pattern = data['pattern']
            self.compiled_pattern = re.compile(self.pattern)
            self.vocab_size = data['vocabsize']
            self.vocab = self._build_vocab()

class MARegexTokenizer(MATokenizer):
   
    def __init__(self, pattern=None , special_tokens = {} , vocab_size = 1000):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.vocab_size = vocab_size 
        self.pattern = self.GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k,v in special_tokens.items()}
        for pair, idx in self.merges:
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
    
    def process_chunk_ids_batch(self,batch_chunks):
            batch_pairs = Counter()
            for chunk_ids in batch_chunks:
                local_pairs = self.get_pairs_parallel(chunk_ids)
                batch_pairs += local_pairs
            return batch_pairs
    
    def update_pairs_forest(self,ids):
            """
            iterate over all the pairs and calculate the frequency for all of them
            """
            pairs = Counter()
            
            chunk_batches = [ids[self.CHUNKING_BATCHSIZE * i : self.CHUNKING_BATCHSIZE * (i + 1)] for i in range((len(ids) + self.CHUNKING_BATCHSIZE - 1) // self.CHUNKING_BATCHSIZE)]
            with concurrent.futures.ProcessPoolExecutor(self.CHUNKING_MAX_WORKER) as executor:
                futures = [executor.submit(self.process_chunk_ids_batch, chunk_ids_batch) for chunk_ids_batch in chunk_batches]
                #print(f"[+] Merging {len(futures)} chunks ...")
                for idx,thread in enumerate( concurrent.futures.as_completed(futures)):
                    local_pairs = thread.result()
                    #print(f"[+] Calculating the Stat {idx+1} / {len(futures)} chunks ...", end="\r")
                    pairs += local_pairs
            return pairs
    
    def process_get_new_ids_batch(self, batch_ids, pair, idx):
        return [self.fast_merge_tokens(chunk_ids, pair, idx) for chunk_ids in batch_ids]

    def get_new_ids(self, ids, pair, idx):
        chunk_batches = [ids[self.MERGE_BATCHING_SIZE * i : self.MERGE_BATCHING_SIZE * (i + 1)] for i in range((len(ids) + self.MERGE_BATCHING_SIZE - 1) // self.MERGE_BATCHING_SIZE)]
        #print(len(chunk_batches))
        new_ids = []
        with concurrent.futures.ProcessPoolExecutor(self.MERGE_BATCHING_MAX_WORKER) as executor:
            futures = [executor.submit(self.process_get_new_ids_batch, chunk_ids, pair, idx) for chunk_ids in chunk_batches]
            for id,thread in enumerate( concurrent.futures.as_completed(futures)):
                new_ids.extend(thread.result())
                #print(f"[+] merge pairs {id+1} / {len(futures)} chunks ...", end="\r")
        return new_ids
    
    def build_bpe_p(self, corpus_path):
        assert self.vocab_size >= 256
        num_merges = self.vocab_size - 256

        print("[+] Read Corpus ...")
        text = self.read_corpus(corpus_path)
        print("[+] Chunking the text using Regex ...")
        text_chunk = re.findall(self.compiled_pattern,text)

        ids = [np.array(list(ch.encode("utf-8"))) for ch in text_chunk]

        self.merges = {}
        

        print(f"[+] do the estimated merges {num_merges} ...")
        for i in range(num_merges):
            start = time.time()
            pairs = Counter()
            pairs = self.update_pairs_forest(ids)
            elapsed_get_pairs = time.time() - start
            pair = pairs.most_common(1)[0][0]
            idx = 256 + i
            start = time.time()
            ids = self.get_new_ids(ids,pair,idx) #[self.fast_merge_tokens(chunk_ids, pair, idx) for chunk_ids in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[int(pair[0])] + self.vocab[int(pair[1])]
            elapsed_merge = time.time() - start
            logger.info(
                f"Merge Done {i+1} / {num_merges} | "
                f"Vocab size: {len(self.vocab)} | "
                f"New Ids: {len(ids)} | "
                f"Work time for Get Pairs: {elapsed_get_pairs:.2f} seconds | "
                f"Work time for Merge: {elapsed_merge:.2f} seconds"
            )

    def build_bpe(self, corpus_path): # type: ignore
        assert self.vocab_size >= 256
        num_merges = self.vocab_size - 256

        print("[+] Read Corpus ...")
        text = self.read_corpus(corpus_path)
        print("[+] Chunking the text using Regex ...")
        text_chunk = re.findall(self.compiled_pattern,text)

        ids = [np.array(list(ch.encode("utf-8"))) for ch in text_chunk]

        self.merges = {}
        
        print(f"[+] do the estimated merges {num_merges} ...")
        for i in range(num_merges):
            start = time.time()
            pairs = Counter()
            for chunk_ids in ids:
                self.get_pairs_counter(chunk_ids,pairs)
            elapsed_get_pairs = time.time() - start
            pair = pairs.most_common(1)[0][0]
            idx = 256 + i
            start = time.time()
            ids = [self.fast_merge_tokens(chunk_ids, pair, idx) for chunk_ids in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            elapsed_merge = time.time() - start
            print(f"""
                  Merge Done {i+1} / {num_merges} 
                  Vocab size: {len(self.vocab)}   
                  New Ids: {len(ids)}    
                  Work time for Get Pairs: {elapsed_get_pairs:.2f} seconds  
                  Work time for Merge: {elapsed_merge:.2f} seconds
                  ______________________________________________________
                  """)

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}            

    def decode(self,ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
            
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            pairs = self.get_pairs(ids,{})
            pair = min(pairs, key = lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge_tokens(ids, pair, idx)
        return ids

    def encode_ordinary(self,text):
        ids = []
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text ,allowed_special="none_raise"):
        special = None
        if allowed_special == 'all':
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == 'none_raise':
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k:v for k,v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            return self.encode_ordinary(text)
        
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

class MAFastBPETokenizer(MATokenizer):
    
    def __init__(self, pattern=None , special_tokens = {} , vocab_size = 1000):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.vocab_size = vocab_size 
        self.pattern = self.GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k,v in special_tokens.items()}
        for pair, idx in self.merges:
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
    
    def get_pairs(self,ids , pairs=None):
        """
        iterate over all the pairs and calculate the frequency for all of them
        """
        pairs = pairs or defaultdict(lambda: {'count': 0, 'tokens': []})
        for i in range(len(ids['token'])-1):
            pair = (ids['token'][i], ids['token'][i+1])
            pairs[pair]['count'] += 1
            pairs[pair]['tokens'].append(ids['token_id'])
        return pairs

    def update_pairs(self,ids , pairs, new_pairs=None):
        """
        iterate over all the pairs and calculate the frequency for all of them
        """
        new_pairs = set() if new_pairs is None else new_pairs
        for i in range(len(ids['token'])-1):
            pair = (ids['token'][i], ids['token'][i+1])
            if pair not in pairs or pair in new_pairs:
                if pair not in pairs:
                    new_pairs.add(pair)
                pairs[pair]['count'] += 1
                pairs[pair]['tokens'].append(ids['token_id'])
        return pairs, new_pairs

    def merge_tokens(self,ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids['token']):
            if ids['token'][i] == pair[0] and i < len(ids['token']) - 1 and ids['token'][i+1] == pair[1]:
                new_ids.append(idx)
                i+=2
            else:
                new_ids.append(ids['token'][i])
                i +=1
        ids['token'] = new_ids
        return ids
    
    def build_bpe(self, corpus_path): # type: ignore
        assert self.vocab_size >= 256
        num_merges = self.vocab_size - 256

        logger.info("[+] Read Corpus ...")
        text = self.read_corpus(corpus_path)
        logger.info("[+] Chunking the text using Regex ...")
        text_chunk = re.findall(self.compiled_pattern,text)

        ids = [{ 'token' : list(ch.encode("utf-8")) , 'merged' : -1  , 'token_id': i} for i,ch in enumerate(text_chunk)]

        self.merges = {}
        pairs = None
        logger.info(f"[+] do the estimated merges {num_merges} ...")
        for idm in range(num_merges):
            start = time.time()
            new_pairs = None
            for i in range(len(ids)):
                if ids[i]['merged'] == -1:
                    pairs = self.get_pairs(ids[i],pairs)
                elif ids[i]['merged'] == 1:
                #else:
                    pairs,new_pairs = self.update_pairs(ids[i], pairs,new_pairs)
                ids[i]['merged'] = 0
            if pairs is None or len(pairs) == 0:
                print(f"[!] No more pairs to merge at iteration {idm+1}. Stopping early.")
                break
            most_common_pair = max(pairs.keys(), key=lambda x: pairs[x]['count'])
            idx = 256 + idm
            elapsed_get_pairs = time.time() - start
            start = time.time()
            for i in pairs[most_common_pair]['tokens']:
                ids[i] = self.merge_tokens(ids[i], most_common_pair, idx)
                ids[i]['merged'] = 1
            
            self.merges[most_common_pair] = idx
            pairs.pop(most_common_pair)
            self.vocab[idx] = self.vocab[most_common_pair[0]] + self.vocab[most_common_pair[1]]
            
            elapsed_merge = time.time() - start
            logger.info(f"""
                  Merge Done {idm+1} / {num_merges} 
                  Vocab size: {len(self.vocab)}   
                  New Ids: {len(ids)}    
                  Work time for Get Pairs: {elapsed_get_pairs:.2f} seconds  
                  Work time for Merge: {elapsed_merge:.2f} seconds
                  ______________________________________________________
                  """)
    
    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}            

    def decode(self,ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
            
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        ids = {'token': list(text_bytes) , 'merged': -1 , 'token_id':0}
        while len(ids['token']) >= 2:
            pairs = self.get_pairs(ids,None)
            pair = min(pairs, key = lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge_tokens(ids, pair, idx)
        return ids['token']

    def encode_ordinary(self,text):
        ids = []
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text ,allowed_special="none_raise"):
        special = None
        if allowed_special == 'all':
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == 'none_raise':
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k:v for k,v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            return self.encode_ordinary(text)
        
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
