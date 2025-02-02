import tiktoken
import os

class Tokenizer:
    def __init__(self,algorithm:str="gpt2",text_file_path:str|None=None):
        if text_file_path is None or not os.path.exists(text_file_path):
            text_file_path = input("Please enter the file path for the text to be tokenized: ")
            if not os.path.exists(text_file_path):
                raise ValueError("Please provide a valid file path")
        with open(text_file_path,"r") as f:
            text = f.read()
        self.encoding = tiktoken.get_encoding(algorithm)
        self.tokens = self.encoding.encode(text,allowed_special={"<|endoftext|>"})

Tokenizer("gpt2","the-verdict.txt")