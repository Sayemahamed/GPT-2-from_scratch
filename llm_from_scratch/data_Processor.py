from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch 
class GPTDataset(Dataset):
    def __init__(self,text:str,tokenizer,max_length:int,stride:int):
        token:list[int]=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        self.data: list[tuple[torch.Tensor, torch.Tensor]]=[
            (
                torch.tensor(token[i:i+max_length],dtype=torch.long),
                torch.tensor(token[i+1:i+1+max_length],dtype=torch.long)
            )for i in range(0,len(token)-max_length,stride)
        ]