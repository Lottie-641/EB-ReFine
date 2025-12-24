import argparse
import os
import sys
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from jellyfish import damerau_levenshtein_distance
from transformers import AutoModel, AutoTokenizer

from neural_network.HistoryDataset import CustomDataset
from neural_network.llamp_multiout import BertMultiOutputClassificationHeads
from preprocessing.log_to_history import Log


def clean_sequence(sequence_str, label2id):
    sequence_list = sequence_str.split(" ")
    end_activity_str = str(label2id["activity"]["ENDactivity"])

    if end_activity_str in sequence_list:
        first_end_index = sequence_list.index(end_activity_str)
        sequence_list = sequence_list[: first_end_index + 1]

    return " ".join(sequence_list)


def remove_word(sentence, word):
    words = sentence.split()
    words = [w for w in words if w != word]
    return " ".join(words)


def pad_list_to_length(seq, target_length, end_id):
    if target_length <= 0:
        return seq

    seq_len = len(seq)
    if seq_len == 0:
        return [0] * target_length

    if seq_len < target_length:
        return seq + [end_id] * (target_length - seq_len)
    if seq_len > target_length:
        return seq[:target_length]
    return seq


def extract_prefix(full_trace, suffix_sequence):
    pad_token = suffix_sequence[-1]
    if pad_token in full_trace:
        effective_full = full_trace[: full_trace.index(pad_token)]
    else:
        effective_full = full_trace

    if pad_token in suffix_sequence:
        effective_suffix = suffix_sequence[: suffix_sequence.index(pad_token)]
    else:
        effective_suffix = suffix_sequence

    prefix_length = len(effective_full) - len(effective_suffix)
    return full_trace[:prefix_length]


def predict_suffix_no_freq(model_output):
    predicted_no_freq = []
    for i in range(len(model_output)):
        pred = model_output[i].argmax(dim=1).cpu().numpy()
        predicted_no_freq.append(str(pred[0]))
    return predicted_no_freq


def predict_suffix_with_freq(
    model_output,
    prefix_sequence,     # list[str]
    trace_frequencies,   # dict{tuple(int): freq}
    label2id,
    beta: float,
    threshold: float,
):
    # Decide fixed padding length
    if len(trace_frequencies) == 0:
        max_len_in_db = 0
    else:
        max_len_in_db = max(len(k) for k in trace_frequencies.keys())
        
    end_id = label2id["activity"]["ENDactivity"]
    # 1) model suffix
    model_suffix_full = predict_suffix_no_freq(model_output)          
    model_suffix_cut = clean_sequence(" ".join(model_suffix_full), label2id).split()
    
    if len(model_suffix_cut) > 0 and model_suffix_cut[-1] == str(end_id):
        model_suffix = model_suffix_cut[:-1]
    else:
        model_suffix = model_suffix_cut

    prefix_ints = [int(x) for x in prefix_sequence]
    suffix_ints = [int(x) for x in model_suffix]
    candidate_trace = prefix_ints + suffix_ints

    # 2) pad candidate before exact match
    padded_candidate = pad_list_to_length(candidate_trace, max_len_in_db, end_id)
    candidate_tuple = tuple(padded_candidate)

    # 3) exact match
    if candidate_tuple in trace_frequencies:
        return model_suffix

    # 4) best match by DL + frequency
    best_trace = None
    best_similarity = -1.0
    best_freq = -1.0
    best_tau = -1.0

    if len(trace_frequencies) == 0 or max_len_in_db == 0:
        return model_suffix

    candidate_str = " ".join(map(str, padded_candidate))
    f_max = max(trace_frequencies.values())

    for hist_trace, freq in trace_frequencies.items():
        hist_list = list(hist_trace)
        padded_hist = pad_list_to_length(hist_list, max_len_in_db, end_id)
        hist_str = " ".join(map(str, padded_hist))

        dl_dist = damerau_levenshtein_distance(candidate_str, hist_str)
        similarity = max(0.0, 1.0 - (dl_dist / max_len_in_db))
        tau = beta * similarity + (1.0 - beta) * (freq / f_max)

        if (
            tau > best_tau
            or (tau == best_tau and similarity > best_similarity)
            or (tau == best_tau and similarity == best_similarity and freq > best_freq)
        ):
            best_tau = tau
            best_similarity = similarity
            best_freq = freq
            best_trace = hist_list

    if best_similarity >= threshold and best_trace is not None and len(best_trace) > len(prefix_ints):
        override_suffix_int = best_trace[len(prefix_ints) :]
        return list(map(str, override_suffix_int))

    return model_suffix


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_next_activities(
    model,
    test_loader,
    label2id,
    trace_frequencies,
    out_path,
    beta,
    threshold,
    device,
):
    model.eval()
    list_dl_distance_no_freq = []
    list_dl_distance_with_freq = []

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as file_dl:
        file_dl.write(
            "Prefix,Predicted_NoFreq,Predicted_WithFreq,Truth,"
            "DL_Score_NoFreq,DL_Score_WithFreq,avg_noFreq,avg_withfreq\n"
        )

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                output = model(input_ids, attention_mask)

                true_sequence = []
                full_trace = []

                for i in range(len(batch["labels"])):
                    true_sequence.append(str(batch["labels"][i].item()))
                    full_trace.append(str(batch["activities"][i].item()))

                prefix_sequence = extract_prefix(full_trace, true_sequence)

                # 1) no freq
                predicted_no_freq = predict_suffix_no_freq(output)
                predicted_no_freq = clean_sequence(" ".join(predicted_no_freq), label2id).split()

                # 2) with freq
                predicted_with_freq = predict_suffix_with_freq(
                    output,
                    prefix_sequence,
                    trace_frequencies,
                    label2id,
                    beta=beta,
                    threshold=threshold,
                )
                predicted_with_freq = clean_sequence(" ".join(predicted_with_freq), label2id).split()

                # DL similarity
                seq_pred_no_freq = clean_sequence(" ".join(predicted_no_freq), label2id)
                seq_pred_with_freq = clean_sequence(" ".join(predicted_with_freq), label2id)
                seq_true = clean_sequence(" ".join(map(str, true_sequence)), label2id)
                seq_prefix = clean_sequence(" ".join(map(str, prefix_sequence)), label2id)

                end_token = str(label2id["activity"]["ENDactivity"])
                seq_pred_no_freq = remove_word(seq_pred_no_freq, end_token)
                seq_pred_with_freq = remove_word(seq_pred_with_freq, end_token)
                seq_true = remove_word(seq_true, end_token)
                seq_prefix = remove_word(seq_prefix, end_token)

                if seq_pred_no_freq == "" and seq_true == "":
                    seq_pred_no_freq = "end"
                    seq_true = "end"
                if seq_pred_with_freq == "":
                    seq_pred_with_freq = "end"

                dl_no = 1 - (
                    damerau_levenshtein_distance(seq_pred_no_freq, seq_true)
                    / max(len(seq_pred_no_freq), len(seq_true))
                )
                dl_w = 1 - (
                    damerau_levenshtein_distance(seq_pred_with_freq, seq_true)
                    / max(len(seq_pred_with_freq), len(seq_true))
                )

                list_dl_distance_no_freq.append(dl_no)
                list_dl_distance_with_freq.append(dl_w)

                file_dl.write(
                    f"{seq_prefix},{seq_pred_no_freq},{seq_pred_with_freq},{seq_true},"
                    f"{dl_no:.3f},{dl_w:.3f},"
                    f"{np.mean(list_dl_distance_no_freq):.3f},{np.mean(list_dl_distance_with_freq):.3f}\n"
                )

    print(f"Avg DL Similarity (No Frequency): {np.mean(list_dl_distance_no_freq):.3f}")
    print(f"Avg DL Similarity (With Frequency): {np.mean(list_dl_distance_with_freq):.3f}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--csv_log", type=str, default="helpdesk")
    p.add_argument("--type", type=str, default="all")

    p.add_argument("--semantic_dir", type=str, default="semantic_data")
    p.add_argument("--models_dir", type=str, default="models")
    p.add_argument("--output_dir", type=str, default="output")

    p.add_argument("--model_name", type=str, default="prajjwal1/bert-medium")
    p.add_argument("--max_len", type=int, default=512)

    # required by you
    p.add_argument("--beta", type=float, default=0.98)
    p.add_argument("--threshold", type=float, default=0.4)

    # runtime
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device-->", device)

    # keep your original behavior
    Log(args.csv_log, args.type)

    base = os.path.join(args.semantic_dir, args.csv_log)

    test = load_pickle(os.path.join(base, f"{args.csv_log}_test_{args.type}.pkl"))
    _y_test = load_pickle(os.path.join(base, f"{args.csv_log}_label_test_{args.type}.pkl"))  # kept, unused
    id2label = load_pickle(os.path.join(base, f"{args.csv_log}_id2label_{args.type}.pkl"))
    label2id = load_pickle(os.path.join(base, f"{args.csv_log}_label2id_{args.type}.pkl"))
    y_train_suffix = load_pickle(os.path.join(base, f"{args.csv_log}_suffix_train_{args.type}.pkl"))
    y_test_prefix = load_pickle(os.path.join(base, f"{args.csv_log}_prefixes_test_{args.type}.pkl"))
    y_test_suffix = load_pickle(os.path.join(base, f"{args.csv_log}_suffix_test_{args.type}.pkl"))
    y_test_activities = load_pickle(os.path.join(base, f"{args.csv_log}_activities_test_{args.type}.pkl"))
    trace_frequencies = load_pickle(os.path.join(base, f"{args.csv_log}_encoded_trace_frequencies_{args.type}.pkl"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation_side="left")
    base_model = AutoModel.from_pretrained(args.model_name)

    test_dataset = CustomDataset(
        test, y_test_suffix, y_test_prefix, y_test_activities, tokenizer, args.max_len
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    output_sizes = [len(id2label["activity"]) for _ in range(len(y_train_suffix))]
    model = BertMultiOutputClassificationHeads(base_model, output_sizes)

    model_path = os.path.join(args.models_dir, f"{args.csv_log}_{args.type}.pth")
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    if args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    out_path = os.path.join(args.output_dir, f"{args.csv_log}_{args.type}.txt")
    predict_next_activities(
        model=model,
        test_loader=test_loader,
        label2id=label2id,
        trace_frequencies=trace_frequencies,
        out_path=out_path,
        beta=args.beta,
        threshold=args.threshold,
        device=device,
    )


if __name__ == "__main__":
    main()
