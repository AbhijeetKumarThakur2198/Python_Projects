# Custom BPE Tokenizer

## Overview
In this project, I undertook the challenge of developing a Byte Pair Encoding (BPE) level tokenizer entirely from scratch, without relying on any pre-existing modules. Motivated by a desire to deepen my understanding of tokenization algorithms and to create a customizable solution, I embarked on a journey of meticulous planning and implementation.

The project commenced with thorough data preprocessing, ensuring the cleanliness and uniformity of input text. Moving forward, I meticulously constructed each component of the BPE tokenizer, from byte-level encoding to the iterative merging of token pairs. Throughout the process, I rigorously tested and refined the tokenizer, striving for optimal performance across diverse datasets and linguistic contexts.

Despite the challenges encountered, the project yielded immense satisfaction and invaluable insights into tokenization algorithms. By eschewing external dependencies and crafting a bespoke solution, I not only expanded my skill set but also contributed to the advancement of natural language processing methodologies.

Let's check our tokenizer code for more details!

## Full Code
Here is the full code of Custom BPE Tokenizer:
```python
class Tokenizer:
    """
    A Byte Pair Encoding (BPE) level tokenizer implemented from scratch.

    Attributes:
        special_tokens (dict): A dictionary to store special tokens and their corresponding indices.
        merges (dict): A dictionary to store merged pairs of indices.
        vocab (dict): A dictionary mapping indices to bytes representing tokens.

    Methods:
        get_stats(ids, counts=None):
            Computes the frequency of pairs of indices in the input list of indices.
        
        merge(ids, pair, idx):
            Merges a pair of indices in the input list of indices and returns the new list.
        
        train(text, vocab_size):
            Trains the tokenizer on the input text and generates the vocabulary.
        
        decode(ids):
            Decodes a list of indices into text using the vocabulary.
        
        encode(text):
            Encodes text into a list of indices using the trained merges.
        
        generate_vocab():
            Generates the vocabulary based on the trained merges and special tokens.
        
        save(file_prefix):
            Saves the tokenizer model to a file with the given file prefix.
        
        load(model_file):
            Loads the tokenizer model from the specified file.
    """

    def __init__(self):
        """
        Initializes the Tokenizer class.
        """
        super().__init__()
        self.special_tokens = {}

    def get_stats(self, ids, counts=None):
        """
        Computes the frequency of pairs of indices in the input list of indices.

        Args:
            ids (list): A list of indices representing tokens.
            counts (dict, optional): A dictionary to store the counts of pairs of indices.

        Returns:
            dict: A dictionary containing the counts of pairs of indices.
        """
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        """
        Merges a pair of indices in the input list of indices and returns the new list.

        Args:
            ids (list): A list of indices representing tokens.
            pair (tuple): A tuple containing the pair of indices to be merged.
            idx (int): The index of the merged pair.

        Returns:
            list: A new list of indices with the specified pair merged.
        """
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
        """
        Trains the tokenizer on the input text and generates the vocabulary.

        Args:
            text (str): The input text to train the tokenizer.
            vocab_size (int): The desired size of the vocabulary.

        Raises:
            AssertionError: If vocab_size is less than 256.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
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
        """
        Decodes a list of indices into text using the vocabulary.

        Args:
            ids (list): A list of indices representing tokens.

        Returns:
            str: The decoded text.
        """
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        """
        Encodes text into a list of indices using the trained merges.

        Args:
            text (str): The input text to be encoded.

        Returns:
            list: A list of indices representing tokens.
        """
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
        """
        Generates the vocabulary based on the trained merges and special tokens.

        Returns:
            dict: A dictionary mapping indices to bytes representing tokens.
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves the tokenizer model to a file with the given file prefix.

        Args:
            file_prefix (str): The prefix for the model file.
        """
        model_file = file_prefix + ".tokenizer"
        with open(model_file, 'w') as f:
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

    def load(self, model_file):
        """
        Loads the tokenizer model from the specified file.

        Args:
            model_file (str): The file containing the tokenizer model.

        Raises:
            AssertionError: If the model file does not end with ".tokenizer".
        """
        assert model_file.endswith(".tokenizer")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
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
```        

## Example Usage Code
Here is the example code to show that how we can use this tokenizer:
```python
from CustomBPETokenizer import Tokenizer

tokenizer = Tokenizer()

data = "Once upon a time"

tokenizer.train(data, 500)
tokenizer.save("a")
tokenizer.load("a.tokenizer")

text = "Once upon a time"

encoded_text = tokenizer.encode(text)
print(encoded_text)

decoded_text = tokenizer.decode(encoded_text)
print(decoded_text)
```

## How To Report Problems
If you encounter any errors or bugs in the code, please create an issue.

## License
This project is under the [MIT License](https://github.com/AbhijeetKumarThakur2198/Python_Projects/tree/main/Projects_List/LICENSE.md) - see the [LICENSE.md](https://github.com/AbhijeetKumarThakur2198/Python_Projects/tree/main/Projects_List/LICENSE.md) file for details.
