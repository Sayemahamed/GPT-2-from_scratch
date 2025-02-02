import tokenize
from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch 
class GPTDataset(Dataset):
    def __init__(self,text:str,tokenizer,max_length:int,stride:int) -> None:
        token:list[int]=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        self.data: list[tuple[torch.Tensor, torch.Tensor]]=[
            (
                torch.tensor(data=token[i:i+max_length],dtype=torch.long),
                torch.tensor(data=token[i+1:i+1+max_length],dtype=torch.long)
            )for i in range(0,len(token)-max_length,stride)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]
    
def create_dataloader(text:str,batch_size:int=4,max_length:int=256,stride:int=128,shuffle:bool=False)->DataLoader:
    tokenizer=tiktoken.get_encoding("gpt2")
    dataset=GPTDataset(text=text,tokenizer=tokenizer,max_length=max_length,stride=stride)
    return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,drop_last=True)