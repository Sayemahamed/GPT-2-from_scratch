from typing import Any
from GPT import model  # assuming these are defined in GPT.py
from data_Processor import create_dataloader
import torch
from torch import nn
import tiktoken
from torch.utils.data import DataLoader
from torch import optim



def generate_text(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int) -> torch.Tensor:
    """
    Autoregressively generate text tokens.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        # Select token with highest probability from the last position.
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

########################################
# Saving and Loading Model Weights
########################################

def save_model(model: nn.Module, path: str = "model_weights.pt"):
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")

def load_model(model: nn.Module, path: str = "model_weights.pt"):
    model.load_state_dict(torch.load(path))
    print(f"Model weights loaded from {path}")

########################################
# Training Function
########################################

def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 5,
    lr: float = 3e-4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model.to(device=device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batch_count = 0
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
        avg_loss: float = total_loss / batch_count
        print(f"Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.4f}")

########################################
# Main: Interactive Loop
########################################

def main():
    # Determine the device and move the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create the tokenizer (using tiktoken for GPT-2).
    tokenizer = tiktoken.get_encoding("gpt2")
    
    while True:
        print("\nSelect an operation mode:")
        mode = input("Enter 'train' to train, 'generate' to generate text, or 'exit' to quit: ").strip().lower()
        if mode == "exit":
            print("Exiting.")
            break
        elif mode == "train":
            print("\n--- Training Mode ---")
            # Ask for training text input.
            text_file = input("Enter the path to a text file for training data (or leave blank to enter text manually): ").strip()
            if text_file:
                try:
                    with open(text_file, "r", encoding="utf-8") as f:
                        training_text = f.read()
                except Exception as e:
                    print(f"Error reading file: {e}")
                    continue
            else:
                training_text = input("Enter training text: ")

            # Create dataloader (adjust parameters as needed).
            dataloader = create_dataloader(
                training_text, batch_size=2, max_length=32, stride=16, shuffle=True
            )
            try:
                epochs = int(input("Enter the number of epochs to train: "))
            except ValueError:
                print("Invalid input for epochs; using default (5).")
                epochs = 5
            train_model(model, dataloader, epochs=epochs, lr=3e-4, device=device)
            
            weight_path = input("Enter a filename to save model weights (default: model_weights.pt): ").strip()
            if not weight_path:
                weight_path = "model_weights.pt"
            save_model(model=model, path=weight_path)
        
        elif mode == "generate":
            print("\n--- Generation Mode ---")
            weight_path = input("Enter the path to load model weights (default: model_weights.pt): ").strip()
            if not weight_path:
                weight_path = "model_weights.pt"
            load_model(model=model, path=weight_path)
            model.eval()
            
            starting_text = input("Enter starting text for generation: ")
            encoded = tokenizer.encode(starting_text)
            idx = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
            try:
                max_new_tokens = int(input("Enter the number of tokens to generate: "))
            except ValueError:
                print("Invalid number, using default of 10.")
                max_new_tokens = 10
            
            # Use the model's context length for slicing.
            context_size = model.context_length if hasattr(model, "context_length") else 128
            out = generate_text(model, idx, max_new_tokens=max_new_tokens, context_size=context_size)
            decoded = tokenizer.decode(out.squeeze(0).tolist())
            print("\nGenerated text:")
            print(decoded)
        else:
            print("Invalid mode. Please enter 'train', 'generate', or 'exit'.")

if __name__ == "__main__":
    main()
