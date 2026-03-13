import torch
import argparse
from transformers import PreTrainedTokenizerFast
from src.model import MolTransformer

def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # In reality, load from checkpoint
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = MolTransformer(vocab_size=tokenizer.vocab_size).to(device)
    model.eval()
    
    start_token = tokenizer.encode('[CLS]')
    
    generated_molecules = []
    
    for _ in range(args.num_molecules):
        inp = torch.tensor([start_token]).to(device)
        
        # Simple greedy decoding for demo
        for _ in range(100):
            out = model(inp, inp)
            next_token = torch.argmax(out[:, -1, :], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            inp = torch.cat([inp, next_token.unsqueeze(0)], dim=1)
            
        smiles = tokenizer.decode(inp.squeeze().tolist(), skip_special_tokens=True)
        generated_molecules.append(smiles)
        
    print(f"Generated {len(generated_molecules)} molecules:")
    for smi in generated_molecules[:10]:
        print(smi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_molecules", type=int, default=10)
    args = parser.parse_args()
    generate(args)