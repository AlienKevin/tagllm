from opencc import OpenCC
cc = OpenCC('t2s')

import random
random.seed(42)

sents = {}

with open('abc-eng-yue/yue.txt', 'r') as f:
    yue = f.read().split('\n')

with open('abc-eng-yue/eng.txt', 'r') as f:
    eng = f.read().split('\n')

with open('wordshk-eng-yue/minus15/yue.txt', 'r') as f:
    yue += f.read().split('\n')

with open('wordshk-eng-yue/minus15/en.txt', 'r') as f:
    eng += f.read().split('\n')

with open('wordshk-eng-yue/plus15/train.yue.txt', 'r') as f:
    yue += f.read().split('\n')

with open('wordshk-eng-yue/plus15/train.en.txt', 'r') as f:
    eng += f.read().split('\n')

sents['eng-yue'] = {}
sents['eng-yue']['train'] = list((eng, yue) for eng, yue in zip(eng, yue) if len(eng) > 0 and len(yue) > 0)

with open('wordshk-eng-yue/plus15/test.yue.txt', 'r') as f:
    yue = f.read().split('\n')

with open('wordshk-eng-yue/plus15/test.en.txt', 'r') as f:
    eng = f.read().split('\n')

sents['eng-yue']['test'] = list((eng, yue) for eng, yue in zip(eng, yue) if len(eng) > 0 and len(yue) > 0)


with open('wmt19-eng-cmn/news-commentary-v14.en-zh.tsv', 'r') as f:
    import csv
    en = []
    zh = []
    reader = csv.reader(f, delimiter='\t')
    for i, line in enumerate(reader):
        if i < 50000:
            if line[0] == '' or line[1] == '':
                continue
            en.append(line[0])
            zh.append(line[1])
        else:
            break

sents['eng-cmn'] = {}

eng_cmn = list(zip(en, zh))
random.shuffle(eng_cmn)
sents['eng-cmn']['train'] = eng_cmn[1500:]
sents['eng-cmn']['test'] = eng_cmn[:1500]


with open('kfcd-cmn-yue/yue.txt', 'r') as f:
    yue = f.read().split('\n')

with open('kfcd-cmn-yue/cmn.txt', 'r') as f:
    cmn = [cc.convert(x) for x in f.read().split('\n')]

cmn_yue = list(zip(cmn, yue))
random.shuffle(cmn_yue)
sents['cmn-yue'] = {}
sents['cmn-yue']['train'] = cmn_yue[1500:]
sents['cmn-yue']['test'] = cmn_yue[:1500]


for key, dataset in sents.items():
    for split, value in dataset.items():
        print(f"Overview for {key} - {split}:")
        print(f"Total sentence pairs: {len(value)}")
        import random
        sampled_pairs = random.sample(value, min(5, len(value)))  # Ensure not to sample more than exists
        print("Sample sentence pairs:")
        for i, pair in enumerate(sampled_pairs):
            print(f"{i+1}: {pair}")
        print("\n")

from datasets import Dataset, DatasetDict

# Create a Huggingface dataset for eng-yue
eng_yue_train = Dataset.from_dict({"translation": [{"eng": eng, "yue": yue} for eng, yue in sents['eng-yue']['train']]})
eng_yue_test = Dataset.from_dict({"translation": [{"eng": eng, "yue": yue} for eng, yue in sents['eng-yue']['test']]})
eng_yue_dataset = DatasetDict({"train": eng_yue_train, "test": eng_yue_test})

# Create a Huggingface dataset for eng-cmn
eng_cmn_train = Dataset.from_dict({"translation": [{"eng": eng, "cmn": cmn} for eng, cmn in sents['eng-cmn']['train']]})
eng_cmn_test = Dataset.from_dict({"translation": [{"eng": eng, "cmn": cmn} for eng, cmn in sents['eng-cmn']['test']]})
eng_cmn_dataset = DatasetDict({"train": eng_cmn_train, "test": eng_cmn_test})

# Create a Huggingface dataset for cmn-yue
cmn_yue_train = Dataset.from_dict({"translation": [{"cmn": cmn, "yue": yue} for cmn, yue in sents['cmn-yue']['train']]})
cmn_yue_test = Dataset.from_dict({"translation": [{"cmn": cmn, "yue": yue} for cmn, yue in sents['cmn-yue']['test']]})
cmn_yue_dataset = DatasetDict({"train": cmn_yue_train, "test": cmn_yue_test})

# Save datasets
eng_yue_dataset.save_to_disk('translations/eng_yue')
eng_cmn_dataset.save_to_disk('translations/eng_cmn')
cmn_yue_dataset.save_to_disk('translations/cmn_yue')

# Load datasets
eng_yue_dataset = DatasetDict.load_from_disk('translations/eng_yue')
eng_cmn_dataset = DatasetDict.load_from_disk('translations/eng_cmn')
cmn_yue_dataset = DatasetDict.load_from_disk('translations/cmn_yue')

# Print the first 5 examples of the training set in a prettier format
print("First 5 examples of the training set:")
for dataset_name, dataset in [("eng_yue", eng_yue_dataset), ("eng_cmn", eng_cmn_dataset), ("cmn_yue", cmn_yue_dataset)]:
    print(f"\n{dataset_name} dataset:")
    for example in dataset['train']['translation'][:5]:
        print(f"- {example}")

# Print the first 5 examples of the test set in a prettier format
print("\nFirst 5 examples of the test set:")
for dataset_name, dataset in [("eng_yue", eng_yue_dataset), ("eng_cmn", eng_cmn_dataset), ("cmn_yue", cmn_yue_dataset)]:
    print(f"\n{dataset_name} dataset:")
    for example in dataset['test']['translation'][:5]:
        print(f"- {example}")
