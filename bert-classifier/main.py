"""
Minimal BERT text classifier with a tiny in-memory dataset.
The script avoids work on import and provides a straightforward training entry point.
"""

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer


@dataclass
class TrainingConfig:
    model_name: str = "bert-base-uncased"
    max_len: int = 64
    batch_size: int = 2
    epochs: int = 6
    lr: float = 2e-5
    test_size: float = 0.2
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name: str) -> BertTokenizer:
    try:
        return BertTokenizer.from_pretrained(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load tokenizer for '{model_name}'. "
            "Download the model first or point to a local path."
        ) from exc


def load_backbone(model_name: str) -> BertModel:
    try:
        return BertModel.from_pretrained(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load backbone weights for '{model_name}'. "
            "Use transformers-cli to download them before running."
        ) from exc


class SimpleTextDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer: BertTokenizer, max_len: int = 64):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class SimpleBERTClassifier(nn.Module):
    def __init__(self, bert_model_name: str, num_labels: int = 2):
        super().__init__()
        self.bert = load_backbone(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_state = outputs.last_hidden_state[:, 0]  # pooler_output is not present on all checkpoints
        cls_state = self.dropout(cls_state)
        return self.classifier(cls_state)


def build_loaders(
    tokenizer: BertTokenizer,
    texts: Sequence[str],
    labels: Sequence[int],
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader]:
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=config.test_size, random_state=config.seed
    )

    train_dataset = SimpleTextDataset(train_texts, train_labels, tokenizer, max_len=config.max_len)
    val_dataset = SimpleTextDataset(val_texts, val_labels, tokenizer, max_len=config.max_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)
    return train_loader, val_loader


def train_epoch(model: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(data_loader))


def eval_model(model: nn.Module, data_loader: DataLoader, device) -> float:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return accuracy_score(all_labels, all_preds)


def run_training(config: TrainingConfig = TrainingConfig()) -> None:
    set_seed(config.seed)

    data = pd.DataFrame(
        {
            "text": [
                "I love this product, it is amazing!",
                "This is the worst service I have ever had.",
                "Absolutely fantastic experience.",
                "I hate this item, very disappointing.",
                "Great quality and fast delivery!",
                "Not satisfied, will not buy again.",
                "Amazing customer support!",
                "Terrible, I want a refund.",
                "Highly recommend this to everyone.",
                "Do not buy this, waste of money.",
            ],
            "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    texts = data["text"].tolist()
    labels = data["label"].tolist()

    tokenizer = load_tokenizer(config.model_name)
    train_loader, val_loader = build_loaders(tokenizer, texts, labels, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleBERTClassifier(config.model_name, num_labels=len(set(labels)))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_acc = eval_model(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    run_training()
