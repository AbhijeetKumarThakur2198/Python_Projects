def Custom_Progress_Bar(iterable, status=False, symbol="#", color=False):
    bar_length = 20
    total = len(iterable)
    previous_block = 0
    counter = 0
    error_bar = ""

    try:
        for item in iterable:
            counter += 1
            progress = counter / total
            block = int(round(bar_length * progress))
            previous_block = block

            if color:
                green_symbol = "\033[92m" + symbol + "\033[0m"
                progress_bar = "[" + green_symbol * block + "-" * (bar_length - block) + "]"
            else:
                progress_bar = "[" + symbol * block + "-" * (bar_length - block) + "]"

            print(f"\r{progress_bar} {progress * 100:.0f}%", end='', flush=True)
            yield item
        print('\n')

        if status:
            if color:
                print("\033[92mProcess completed!\033[0m\n")
            else:
                print("Process completed!\n")

    except Exception as e:
        if color==True:
            red_symbol = "\033[91m" + symbol + "\033[0m"
            error_bar = "[" + red_symbol * previous_block + "-" * (bar_length - previous_block) + "]"
        else:
        	error_bar = "[" + symbol * previous_block + "-" * (bar_length - previous_block) + "]"
        	
        print(f"\r{error_bar}", end='', flush=True)
        print("\n")
       
        if status:
            if color:        	
                print(f"\033[91mError: {e}\033[0m\n")
            else:
            	print(f"Error: {e}\n")

class Custom_(BPE)_Tokenizer:
    def __init__(self):
        super().__init__()                
        self.special_tokens = {}        
        
    def get_stats(self, ids, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text, vocab_size):
	    assert vocab_size >= 256
	    num_merges = vocab_size - 256
	    text_bytes = text.encode("utf-8")
	    ids = list(text_bytes)
	    merges = {}	    
	    vocab = {idx: bytes([idx]) for idx in range(256)}	    	    	
	    for i in Custom_Progress_Bar(range(num_merges), color=True, status=True):
	        stats = self.get_stats(ids)
	        if not stats: 
	            break
	        pair = max(stats, key=stats.get)
	        idx = 256 + i
	        ids = self.merge(ids, pair, idx)
	        merges[pair] = idx
	        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]	     	        
	          
	    self.merges = merges
	    self.vocab = vocab
	
    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
        return ids

    def generate_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        model_file = file_prefix + ".tokenizer"
        with open(model_file, 'w') as f:
            f.write("! WARRANING DON'T CHANGE THIS FILE YOU MAY GET ERROR !\n")       
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")        
        
    def load(self, model_file):
        assert model_file.endswith(".tokenizer")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            f.readline().strip()            
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self.generate_vocab()
