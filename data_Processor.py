import tokenize
from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch


class GPTDataset(Dataset):
    def __init__(self, text: str, tokenizer, max_length: int, stride: int) -> None:
        """
        Parameters:
            text (str): the text to process
            tokenizer (tiktoken.Encoding): the tokenizer to use
            max_length (int): the maximum length of the input sequence
            stride (int): the stride to use when splitting the input text into sequences
        """
        token: list[int] = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        self.data: list[tuple[torch.Tensor, torch.Tensor]] = [
            (
                torch.tensor(data=token[i : i + max_length], dtype=torch.long),
                torch.tensor(data=token[i + 1 : i + 1 + max_length], dtype=torch.long),
            )
            for i in range(0, len(token) - max_length, stride)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]


def create_dataloader(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = False,
) -> DataLoader:
    """
    Creates a DataLoader for a language model dataset.

    Args:
        text (str): The input text to be tokenized and split into sequences.
        batch_size (int, optional): The number of samples per batch. Defaults to 4.
        max_length (int, optional): The maximum length of each sequence. Defaults to 256.
        stride (int, optional): The step size between the start of each sequence. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the dataset before each epoch. Defaults to False.

    Returns:
        DataLoader: A DataLoader object providing batches of input-output sequence pairs.
    """

    tokenizer: tiktoken.Encoding = tiktoken.get_encoding(encoding_name="gpt2")
    dataset = GPTDataset(
        text=text, tokenizer=tokenizer, max_length=max_length, stride=stride
    )
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
