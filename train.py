import argparse
import os
import time
import random
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split

from neural_network.HistoryDataset import CustomDataset
from neural_network.llamp_multiout import BertMultiOutputClassificationHeads
from preprocessing.log_to_history import Log


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def train_fn(model, train_loader, optimizer, device, criterion, label_keys):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]  # expected dict-like: labels[key] -> tensor

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)  # list/tuple aligned with label_keys

        loss = 0.0
        for i, key in enumerate(label_keys):
            loss = loss + criterion[key](outputs[i].to(device), labels[key].to(device))

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(train_loader))


def evaluate_fn(model, data_loader, criterion, device, label_keys):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)

            loss = 0.0
            for i, key in enumerate(label_keys):
                loss = loss + criterion[key](outputs[i].to(device), labels[key].to(device))

            total_loss += float(loss.item())

    return total_loss / max(1, len(data_loader))


def train_llm(model, train_loader, val_loader, optimizer, epochs, criterion, device, label_keys, patience=10):
    best_valid_loss = float("inf")
    best_state = None
    early_stop_counter = 0

    for epoch in range(epochs):
        train_loss = train_fn(model, train_loader, optimizer, device, criterion, label_keys)
        valid_loss = evaluate_fn(model, val_loader, criterion, device, label_keys)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {valid_loss:.4f}")

        if early_stop_counter >= patience:
            print(f"Validation loss hasn't improved for {patience} epochs. Early stopping...")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def parse_args():
    p = argparse.ArgumentParser()

    # dataset / paths
    p.add_argument("--csv_log", type=str, default="helpdesk", help="dataset/log name (e.g., helpdesk)")
    p.add_argument("--type", type=str, default="all", help="TYPE (e.g., all)")
    p.add_argument("--semantic_dir", type=str, default="semantic_data", help="base folder of semantic_data")

    # model / training
    p.add_argument("--model_name", type=str, default="prajjwal1/bert-medium")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    # runtime
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true", help="use DataParallel when multiple GPUs are available")

    # outputs
    p.add_argument("--models_dir", type=str, default="models")
    p.add_argument("--output_dir", type=str, default="output")
    p.add_argument("--early_stop_patience", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device-->", device)


    Log(args.csv_log, args.type)

    base = os.path.join(args.semantic_dir, args.csv_log)

    id2label_path = os.path.join(base, f"{args.csv_log}_id2label_{args.type}.pkl")
    label2id_path = os.path.join(base, f"{args.csv_log}_label2id_{args.type}.pkl")
    train_path = os.path.join(base, f"{args.csv_log}_train_{args.type}.pkl")
    y_train_path = os.path.join(base, f"{args.csv_log}_label_train_{args.type}.pkl")
    y_train_suffix_path = os.path.join(base, f"{args.csv_log}_suffix_train_{args.type}.pkl")

    id2label = load_pickle(id2label_path)
    _label2id = load_pickle(label2id_path)  
    train = load_pickle(train_path)
    _y_train = load_pickle(y_train_path)  
    y_train_suffix = load_pickle(y_train_suffix_path)

    # split inputs
    train_input, val_input = train_test_split(train, test_size=0.2, random_state=42)

    train_label = {}
    val_label = {}
    for key in y_train_suffix.keys():
        train_label[key], val_label[key] = train_test_split(
            y_train_suffix[key], test_size=0.2, random_state=42
        )

    # tokenizer / backbone
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation_side="left")
    backbone = AutoModel.from_pretrained(args.model_name)

    label_keys = list(train_label.keys())

    output_sizes = [len(id2label["activity"]) for _ in label_keys]

    train_dataset = CustomDataset(train_input, train_label, tokenizer, args.max_len)
    val_dataset = CustomDataset(val_input, val_label, tokenizer, args.max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = BertMultiOutputClassificationHeads(backbone, output_sizes).to(device)

    # multi-GPU (DataParallel)
    if args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # criterion per head
    criterion = {k: torch.nn.CrossEntropyLoss() for k in label_keys}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_time = time.time()
    model = train_llm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        criterion=criterion,
        device=device,
        label_keys=label_keys,
        patience=args.early_stop_patience,
    )
    exec_time = time.time() - start_time

    os.makedirs(args.models_dir, exist_ok=True)
    save_path = os.path.join(args.models_dir, f"{args.csv_log}_{args.type}.pth")

    # DataParallel needs .module for clean state_dict
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state_dict, save_path)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.csv_log}_{args.type}.txt"), "w") as f:
        f.write(str(exec_time))

    print(f"Saved model to: {save_path}")
    print(f"Execution time (s): {exec_time:.2f}")


if __name__ == "__main__":
    main()