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

with open ('pos_Meta-Llama-3-8B-qlora-pos.jsonl', 'r') as f:
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

print('accuracy:', round(results["accuracy"], 3))
print('macro avg f1:', round(results['macro avg']['f1-score'], 3))
print('weighted avg f1:', round(results['weighted avg']['f1-score'], 3))
