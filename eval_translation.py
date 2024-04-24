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

with open ('translations_Meta-Llama-3-8B.jsonl', 'r') as f:
    translations = [json.loads(line) for line in f]

sacrebleu = evaluate.load("sacrebleu")

references = []
predictions = []

for ref, pred in zip(get_dataset(), translations):
    references.append(ref['target'])
    predictions.append(pred['sents'][-1])

results = sacrebleu.compute(predictions=predictions,
                             references=references, tokenize='zh')

print('BLEU:', round(results["score"], 1))

chrf = evaluate.load("chrf")

results = chrf.compute(predictions=predictions,
                             references=references, word_order=2)

print('ChRF++:', round(results["score"], 1))
