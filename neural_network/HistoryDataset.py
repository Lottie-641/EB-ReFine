import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, texts, labels, prefixes, activities, tokenizer, max_len):
        self.texts = texts
        self.labels = {}
        for v in labels:
            self.labels[v] = labels[v]
            
        self.activities = {}
        for i in activities:
            self.activities[i] = activities[i]
            
        self.prefixes = {}
        for i in prefixes:
            self.prefixes[i] = prefixes[i]

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = {}
        for v in self.labels:
            label[v] = self.labels[v][idx]
            
        activity = {}
        for i in self.activities:
            activity[i] = self.activities[i][idx]

        prefix = {}
        for i in self.prefixes:
            prefix[i] = self.prefixes[i][idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label,
            'activities': activity,
            'prefixes': prefix
        }