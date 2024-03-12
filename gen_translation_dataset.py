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

# Create a Huggingface dataset for eng-yue and save to parquet
eng_yue_train = Dataset.from_dict({"translation": [{"eng": eng, "yue": yue} for eng, yue in sents['eng-yue']['train']]})
eng_yue_test = Dataset.from_dict({"translation": [{"eng": eng, "yue": yue} for eng, yue in sents['eng-yue']['test']]})
import os

# Create directories for each language pair if they do not exist
language_pairs = ['eng-yue', 'eng-cmn', 'cmn-yue']
for pair in language_pairs:
    os.makedirs(f'translations/{pair}', exist_ok=True)

eng_yue_train.to_parquet('translations/eng-yue/train-00000-of-00001.parquet')
eng_yue_test.to_parquet('translations/eng-yue/test-00000-of-00001.parquet')

# Create a Huggingface dataset for eng-cmn and save to parquet
eng_cmn_train = Dataset.from_dict({"translation": [{"eng": eng, "cmn": cmn} for eng, cmn in sents['eng-cmn']['train']]})
eng_cmn_test = Dataset.from_dict({"translation": [{"eng": eng, "cmn": cmn} for eng, cmn in sents['eng-cmn']['test']]})
eng_cmn_train.to_parquet('translations/eng-cmn/train-00000-of-00001.parquet')
eng_cmn_test.to_parquet('translations/eng-cmn/test-00000-of-00001.parquet')

# Create a Huggingface dataset for cmn-yue and save to parquet
cmn_yue_train = Dataset.from_dict({"translation": [{"cmn": cmn, "yue": yue} for cmn, yue in sents['cmn-yue']['train']]})
cmn_yue_test = Dataset.from_dict({"translation": [{"cmn": cmn, "yue": yue} for cmn, yue in sents['cmn-yue']['test']]})
cmn_yue_train.to_parquet('translations/cmn-yue/train-00000-of-00001.parquet')
cmn_yue_test.to_parquet('translations/cmn-yue/test-00000-of-00001.parquet')

datacard_yaml = f"""
---
language:
- eng
- yue
- cmn
task_categories:
- translation
task_ids: []
config_names:
- eng-yue
- eng-cmn
- cmn-yue
dataset_info:
- config_name: eng-yue
  features:
  - name: translation
    dtype:
      translation:
        languages:
        - eng
        - yue
  splits:
  - name: test
    num_examples: {len(sents['eng-yue']['test'])}
  - name: train
    num_examples: {len(sents['eng-yue']['train'])}
- config_name: eng-cmn
  features:
  - name: translation
    dtype:
      translation:
        languages:
        - eng
        - cmn
  splits:
  - name: test
    num_examples: {len(sents['eng-cmn']['test'])}
  - name: train
    num_examples: {len(sents['eng-cmn']['train'])}
- config_name: cmn-yue
  features:
  - name: translation
    dtype:
      translation:
        languages:
        - cmn
        - yue
  splits:
  - name: test
    num_examples: {len(sents['cmn-yue']['test'])}
  - name: train
    num_examples: {len(sents['cmn-yue']['train'])}
configs:
- config_name: eng-yue
  data_files:
  - split: test
    path: eng-yue/test-*
  - split: train
    path: eng-yue/train-*
- config_name: eng-cmn
  data_files:
  - split: test
    path: eng-cmn/test-*
  - split: train
    path: eng-cmn/train-*
- config_name: cmn-yue
  data_files:
  - split: test
    path: cmn-yue/test-*
  - split: train
    path: cmn-yue/train-*
---
"""
print(datacard_yaml)
