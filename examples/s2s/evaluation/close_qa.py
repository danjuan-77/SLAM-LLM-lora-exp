import argparse
import unicodedata
import string
import json
import re

def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation, articles, extra whitespace, and special tokens."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def remove_special_tokens(text):
        return re.sub(r'<\|.*?\|>', '', text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(remove_special_tokens(lower(s)))))


def exact_match(pred: str, gt: str) -> bool:
    """Check if normalized prediction exactly matches the ground truth."""
    return normalize_text(pred) == normalize_text(gt)


def exist_match(pred: str, gt: str) -> bool:
    """Check if normalized ground truth is contained within normalized prediction."""
    pred_norm = normalize_text(pred)
    # gt_norm = normalize_text(gt)
    gt_parts = [normalize_text(part.strip()) for part in gt.split(',')]
    return all(part in pred_norm for part in gt_parts)
    # return gt_norm in pred_norm


def read_tsv(path: str) -> dict:
    """Read TSV file into a dictionary: {key: value}"""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            key, val = parts
            data[key] = val
    return data

def read_jsonl(path: str) -> tuple:
    """
    Reads a JSONL file and returns two dicts: preds, gts
    Assumes each line contains {"predict": ..., "label": ...}
    """
    preds, gts = {}, {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                key = f"sample_{i}"
                preds[key] = data["predict"]
                gts[key] = data["label"]
            except Exception as e:
                print(f"[Error] Failed to parse line {i}: {e}")
    return preds, gts

def evaluate(pred_file: str, gt_file: str, use_exist_match: bool = False, file_format: str = "tsv", show_mixmatch: bool = False):
    if file_format == "tsv":
        preds = read_tsv(pred_file)
        gts = read_tsv(gt_file)
    elif file_format == "jsonl":
        preds, gts = read_jsonl(pred_file)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    total = 0
    correct = 0
    mismatches = []

    for key in gts:
        if key not in preds:
            print(f"[Warning] Missing prediction for key: {key}")
            continue
        total += 1
        pred = preds[key]
        gt_text = gts[key]
        gt_list = [s.strip() for s in gt_text.split("|||") if s.strip()]  # split refs

        match = any(
            exist_match(pred, gt) if use_exist_match else exact_match(pred, gt)
            for gt in gt_list
        )

        if match:
            correct += 1
        else:
            mismatches.append((key, gt_text, pred))

    accuracy = correct / total if total > 0 else 0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    if mismatches and show_mixmatch:
        print(f"\n[Examples of Incorrect Predictions] ({len(mismatches)} shown)")
        for key, gt, pred in mismatches[:10]:
            print(f"{key}\n  GTs : {gt}\n  Pred: {pred}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions using exact or existence match.")
    parser.add_argument('--pred', type=str, required=True, help='Path to prediction TSV file.')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth TSV file.')
    parser.add_argument('--exist', action='store_true', help='Use existence match instead of exact match.')
    parser.add_argument('--format', type=str, default='tsv', choices=['tsv', 'jsonl'], help='File format of the input files.')
    parser.add_argument('--show-mixmatch', action='store_true', help='Show mixed match examples.')

    args = parser.parse_args()
    evaluate(args.pred, args.gt, use_exist_match=args.exist, file_format=args.format, show_mixmatch=args.show_mixmatch)

if __name__ == "__main__":
    main()