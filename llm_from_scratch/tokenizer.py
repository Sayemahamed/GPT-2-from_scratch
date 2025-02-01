import tiktoken
import os

class Tokenizer:
    def __init__(self,algorithm:str="gpt2",path:str|None=None):