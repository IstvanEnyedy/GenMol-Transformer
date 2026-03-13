# GenMol-Transformer: De Novo Molecular Design with Generative AI

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Status](https://img.shields.io/badge/status-active-success)

## Overview

**GenMol-Transformer** is a state-of-the-art generative framework designed to accelerate drug discovery. By leveraging a Transformer-based architecture (similar to GPT), this model learns the chemical language of molecules (SMILES representation) to generate novel, synthetically viable drug-like structures.

This repository implements a **Generative Pre-trained Transformer (GPT)** specifically fine-tuned for chemical space exploration, enabling the generation of molecules with optimized physicochemical properties (QED, LogP, SA score).

## Key Features

- **Transformer Architecture:** Multi-head self-attention mechanism to capture long-range dependencies in molecular strings.
- **SMILES Tokenization:** Custom regex-based tokenizer optimized for chemical syntax.
- **De Novo Generation:** Generate valid, unique, and novel molecular structures from scratch.
- **Property Optimization:** (Coming Soon) Reinforcement Learning (RL) loop for targeted property optimization.
- **Integration:** Built with PyTorch and RDKit for seamless integration into modern CADD workflows.

## Architecture

The model processes SMILES strings as sequences of tokens.
`mermaid
graph LR
    A[SMILES Dataset] --> B(Tokenizer)
    B --> C{Transformer Encoder-Decoder}
    C --> D[Logits / Next Token Prediction]
    D --> E[Novel SMILES]
`

## Installation

`ash
git clone https://github.com/IstvanEnyedy/GenMol-Transformer.git
cd GenMol-Transformer
pip install -r requirements.txt
`

## Usage

### 1. Training
Train the model on a dataset of SMILES strings (e.g., ChEMBL, ZINC).

`ash
python train.py --data_path data/chembl_corpus.txt --epochs 10
`

### 2. Generation
Generate new molecules after training.

`ash
python generate.py --model_path checkpoints/best_model.pt --num_molecules 100
`

## Requirements

- Python 3.8+
- PyTorch
- RDKit
- Transformers
- NumPy
- Pandas

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
**Developed by Istvan Enyedy, PhD**