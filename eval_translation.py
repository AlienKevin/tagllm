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

with open ('translations_Meta-Llama-3-8B-tagllm-translation-1-reserved-unsloth.jsonl', 'r') as f:
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

def print_scores(predictions, references):
    sacrebleu = evaluate.load("sacrebleu")
    results = sacrebleu.compute(predictions=predictions,
                             references=references, tokenize='zh')
    print('BLEU:', round(results["score"], 1))

    chrf = evaluate.load("chrf")
    results = chrf.compute(predictions=predictions,
                                references=references, word_order=2)
    print('ChRF++:', round(results["score"], 1))

print('Seen language pair:')
print_scores(seen_predictions, seen_references)
print('Unseen language pair:')
print_scores(unseen_predictions, unseen_references)
print('All language pairs:')
print_scores(seen_predictions + unseen_predictions, seen_references + unseen_references)
