from datasets import load_dataset
import evaluate
import json

def get_dataset():
    dataset = load_dataset("universal_dependencies", "yue_hk")
    train_dataset = dataset["test"]

    def preprocess_function(example):
        example["tags"] = [train_dataset.features["upos"].feature.int2str(tag).lower() for tag in example["upos"]]
        return example
    
    train_dataset = train_dataset.map(preprocess_function, remove_columns=
        ['idx', 'text', 'tokens', 'lemmas', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc'])
    return train_dataset

def eval(file_name):
    with open (file_name, 'r') as f:
        tags = [json.loads(line)['sents'][-1].split() for line in f]

    poseval = evaluate.load("poseval")

    references = []
    predictions = []

    for ref, pred in zip(get_dataset(), tags):
        # if len(pred) < len(ref['tags']):
        #     print(ref['tags'])
        #     print(pred)
        references.append(ref['tags'])
        if len(pred) > len(ref['tags']):
            # truncate if longer
            pred = pred[:len(ref['tags'])]
        elif len(pred) < len(ref['tags']):
            # pad if shorter
            pred += ['X'] * (len(ref['tags']) - len(pred))
        predictions.append(pred)

    results = poseval.compute(predictions=predictions, references=references, zero_division=0)

    accuracy = round(results["accuracy"], 3)
    macro_avg_f1 = round(results['macro avg']['f1-score'], 3)
    macro_avg_precision = round(results["macro avg"]["precision"], 3)
    macro_avg_recall = round(results["macro avg"]["recall"], 3)
    weighted_avg_f1 = round(results['weighted avg']['f1-score'], 3)
    weighted_avg_precision = round(results["weighted avg"]["precision"], 3)
    weighted_avg_recall = round(results["weighted avg"]["recall"], 3)

    return accuracy, macro_avg_f1, macro_avg_precision, macro_avg_recall, weighted_avg_f1, weighted_avg_precision, weighted_avg_recall

import glob

# Find all jsonl files starting with 'pos_'
pos_files = glob.glob('pos_*.jsonl')

import csv

# Evaluate each found file and write results to a CSV file
with open('pos_evaluation_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Config', 'Accuracy', 'Macro Avg F1', 'Macro Avg Precision', 'Macro Avg Recall', 'Weighted Avg F1', 'Weighted Avg Precision', 'Weighted Avg Recall'])
    
    for file in pos_files:
        accuracy, macro_avg_f1, macro_avg_precision, macro_avg_recall, weighted_avg_f1, weighted_avg_precision, weighted_avg_recall = eval(file)
        writer.writerow([file, accuracy, macro_avg_f1, macro_avg_precision, macro_avg_recall, weighted_avg_f1, weighted_avg_precision, weighted_avg_recall])
