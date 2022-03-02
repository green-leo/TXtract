from transformers import AutoModel, AutoTokenizer # Thư viện BERT

# Load vinai phobert-base
bert = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
bert.save_pretrained("./phobert-base")
tokenizer.save_pretrained("./phobert-base")