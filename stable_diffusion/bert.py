import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from typing import Union, List


class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased')
        self.m = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, texts) -> torch.Tensor:
        tokens = self.tokenize(texts)
        tokens = {k: v.to(self.m.device) for k, v in tokens.items()}
        with torch.no_grad():
            out = self.m(**tokens)
            return out.last_hidden_state

    def tokenize(self, texts:Union[List[str], str], max_length:int = 64) -> dict:
        if isinstance(texts, str):
            texts = [texts]
        return self.tokenizer(texts, max_length = max_length, padding='max_length', truncation=True, return_tensors='pt')
    