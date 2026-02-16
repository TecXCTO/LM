from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer

class StreamingBPEPipeline(IterableDataset):
    def __init__(self, file_path, tokenizer_name="gpt2", max_seq_len=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.file_path = file_path
        self.max_seq_len = max_seq_len

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = self.tokenizer.encode(line)
                # Chunking tokens into max_seq_len for training
                for i in range(0, len(tokens) - self.max_seq_len, self.max_seq_len):
                    chunk = tokens[i : i + self.max_seq_len + 1]
                    yield torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

# Usage
# dataset = StreamingBPEPipeline("massive_corpus.txt")
# loader = DataLoader(dataset, batch_size=8)
