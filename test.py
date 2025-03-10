from data_Processor import create_dataloader

with open("the-verdict.txt") as f:
    text = f.read()
    loader = create_dataloader(text)

for input, target in loader:
    print("input:", input, "target:", target)
