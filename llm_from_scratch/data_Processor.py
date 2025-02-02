import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, text, max_length, stride):
        # Initialize the tokenizer and encode the text.
        tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2")
        tokens: list[int] = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
        self.samples = []
        for i in range(0, len(tokens) - max_length, stride):
            input_ids = tokens[i:i + max_length]
            target_ids = tokens[i + 1:i + max_length + 1]
            self.samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_dataloader(text, batch_size=4, max_length=256, stride=128, shuffle=True):
    dataset = GPTDataset(text, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

# Example usage:
if __name__ == "__main__":
    with open("the-verdict.txt", "r") as f:
        sample_text = f.read()
    dataloader = get_dataloader(sample_text, batch_size=4, max_length=10, stride=5)
    for input_ids, target_ids in dataloader:
        print("Input IDs:\n", input_ids)
        print("Target IDs:\n", target_ids)
        # break  # Just show one batch for demonstration.
