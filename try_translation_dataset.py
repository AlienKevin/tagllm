from datasets import load_dataset

# Load datasets
eng_yue_dataset = load_dataset("AlienKevin/yue-cmn-eng", "eng_yue")
eng_cmn_dataset = load_dataset("AlienKevin/yue-cmn-eng", "eng_cmn")
cmn_yue_dataset = load_dataset("AlienKevin/yue-cmn-eng", "cmn_yue")

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
