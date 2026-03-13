import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from src.model import MolTransformer, generate_square_subsequent_mask
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock dataset and tokenizer for demonstration
class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_len=100):
        self.data = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"] * 100
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        tokens = self.tokenizer.encode(smiles, padding='max_length', truncation=True, max_length=self.max_len)
        return torch.tensor(tokens)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on {device}")

    # Mock tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gpt2') 
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    dataset = MoleculeDataset(args.data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = MolTransformer(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            src = batch[:, :-1]
            tgt = batch[:, 1:]
            
            tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            output = model(src, tgt)
            loss = criterion(output.view(-1, tokenizer.vocab_size), tgt.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), "checkpoints/best_model.pt")
    logger.info("Model saved!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/chembl.txt")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train(args)