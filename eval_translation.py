from datasets import load_dataset, interleave_datasets
import evaluate
import json

def get_dataset():
    lm_datasets_test = []

    lang_datasets = ["eng-yue", "cmn-yue"]
    lang_pairs = ["eng-yue", "yue-cmn"]

    for i, lang_dataset in enumerate(lang_datasets):
        source_lang, target_lang = lang_pairs[i].split("-")

        def preprocess_eval(examples):
            examples["target"] = [example[target_lang] for example in examples["translation"]]
            del examples['translation']
            return examples

        lm_dataset = load_dataset("AlienKevin/yue-cmn-eng", lang_dataset)
        lm_dataset_test = lm_dataset["test"].map(preprocess_eval, batched=True)
        lm_datasets_test.append(lm_dataset_test)

    eval_dataset = interleave_datasets(lm_datasets_test)
    return eval_dataset

import glob
import csv

def evaluate_translation(file_name):
    with open(file_name, 'r') as f:
        translations = [json.loads(line) for line in f]

    seen_references = []
    seen_predictions = []

    unseen_references = []
    unseen_predictions = []

    for i, (ref, pred) in enumerate(zip(get_dataset(), translations)):
        if i % 2 == 0:
            seen_references.append(ref['target'])
            seen_predictions.append(pred['sents'][-1])
        else:
            unseen_references.append(ref['target'])
            unseen_predictions.append(pred['sents'][-1])

    def compute_scores(predictions, references):
        sacrebleu = evaluate.load("sacrebleu")
        bleu_results = sacrebleu.compute(predictions=predictions, references=references, tokenize='zh')
        chrf = evaluate.load("chrf")
        chrf_results = chrf.compute(predictions=predictions, references=references, word_order=2)
        return round(bleu_results["score"], 1), round(chrf_results["score"], 1)

    seen_bleu, seen_chrf = compute_scores(seen_predictions, seen_references)
    unseen_bleu, unseen_chrf = compute_scores(unseen_predictions, unseen_references)
    all_bleu, all_chrf = compute_scores(seen_predictions + unseen_predictions, seen_references + unseen_references)

    return seen_bleu, seen_chrf, unseen_bleu, unseen_chrf, all_bleu, all_chrf

translation_files = glob.glob('experiment_results/translations_*.jsonl')

with open('experiment_results/translation_evaluation_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Config', 'Seen BLEU', 'Seen ChrF++', 'Unseen BLEU', 'Unseen ChrF++', 'All BLEU', 'All ChrF++'])
    
    for file_name in translation_files:
        seen_bleu, seen_chrf, unseen_bleu, unseen_chrf, all_bleu, all_chrf = evaluate_translation(file_name)
        writer.writerow([file_name, seen_bleu, seen_chrf, unseen_bleu, unseen_chrf, all_bleu, all_chrf])
