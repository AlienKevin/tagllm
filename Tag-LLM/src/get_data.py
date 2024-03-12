from datasets import load_dataset, interleave_datasets


def get_dataset(task_name, num_existing_tokens, tag_name_dict, num_token_per_tag, use_domain_tag, use_function_tag, regression, freeze, is_7b):
    domain_tags = []
    
    if task_name == "Translate":
        lm_datasets_train = []
        lm_datasets_test = []
        
        for lang_dataset in ["eng-yue", "eng-cmn"]:
            lm_dataset = load_dataset("AlienKevin/yue-cmn-eng", lang_dataset)
            lm_dataset_train = lm_dataset["train"]
            lm_dataset_test = lm_dataset["test"]
        
            source_lang = lang_dataset[:2]
            target_lang = lang_dataset[-2:]

            def preprocess_function(examples):
                examples["task"] = [lang_dataset for _ in examples["translation"]]
                examples["input"] = [example[source_lang] for example in examples["translation"]]
                examples["output"] = [example[target_lang] for example in examples["translation"]]
                examples["formulation"] = ["# # Input: <" + source_lang.upper() + "> <input>. \n# # Output: <" + target_lang.upper() + "> <Translate> <output>" for _ in examples["translation"]]
                examples["regression"] = [False for _ in examples["translation"]]
                examples["regression_dim"] = [-1 for _ in examples["translation"]]

                examples["task"] += [target_lang + "-" + source_lang for _ in examples["translation"]]
                examples["input"] += [example[target_lang] for example in examples["translation"]]
                examples["output"] += [example[source_lang] for example in examples["translation"]]
                examples["formulation"] += ["# # Input: <" + target_lang.upper() + "> <input>. \n# # Output: <" + source_lang.upper() + "> <Translate> <output>" for _ in examples["translation"]]
                examples["regression"] += [False for _ in examples["translation"]]
                examples["regression_dim"] += [-1 for _ in examples["translation"]]

                del examples['translation']
                return examples

            lm_dataset_train = lm_dataset_train.map(preprocess_function, batched=True)
            lm_dataset_test = lm_dataset_test.map(preprocess_function, batched=True)
            lm_datasets_train.append(lm_dataset_train)
            lm_datasets_test.append(lm_dataset_test)
        
        train_dataset = interleave_datasets(lm_datasets_train)
        eval_dataset = interleave_datasets(lm_datasets_test)
        
        existing_tokens = ["<ENG>", "<YUE>", "<CMN>"] if use_domain_tag else []
        for tname in existing_tokens:
            idx = tag_name_dict[tname].find(">")
            domain_tags.append(int(tag_name_dict[tname][5:idx]))    
        
        tags_to_update = ["<Translate>"]
        for tname in tags_to_update:
            tag_name_dict[tname] = "".join(["<TAG " + str(i) + ">" for i in range(num_existing_tokens, num_existing_tokens + num_token_per_tag)])
            num_existing_tokens += num_token_per_tag
        
        num_new_tokens = len(tags_to_update) * num_token_per_tag
        tags_to_update = existing_tokens + tags_to_update
        
        
    elif task_name == "Language":
        lm_datasets_train = []
        lm_datasets_test = []
        
        single_lang = ["eng","yue","cmn"]

        for i, lang_dataset in enumerate(single_lang):

            lm_dataset = load_dataset("AlienKevin/yue-cmn-eng", lang_dataset)
            lm_dataset_train = lm_dataset["train"]
            lm_dataset_test = lm_dataset["test"]
        
            target_lang = single_lang[i]

            def preprocess_function(examples):
                examples["task"] = ["Generation" for _ in examples["translation"]]
                examples["input"] = [example[target_lang] for example in examples["translation"]]
                examples["output"] = [example[target_lang] for example in examples["translation"]]
                examples["formulation"] = ["<" + target_lang.upper() + "> <input>" for _ in examples["translation"]]
                examples["regression"] = [False for _ in examples["translation"]]
                examples["regression_dim"] = [-1 for _ in examples["translation"]]

                del examples['translation']
                return examples

            lm_dataset_train = lm_dataset_train.map(preprocess_function, batched=True)
            lm_dataset_test = lm_dataset_test.map(preprocess_function, batched=True)
            lm_datasets_train.append(lm_dataset_train)
            lm_datasets_test.append(lm_dataset_test)

        train_dataset = interleave_datasets(lm_datasets_train)
        eval_dataset = interleave_datasets(lm_datasets_test)

        tags_to_update = ["<ENG>","<YUE>","<CMN>"]
        for tname in tags_to_update:
            domain_tags.append(num_existing_tokens)
            tag_name_dict[tname] = "".join(["<TAG " + str(i) + ">" for i in range(num_existing_tokens, num_existing_tokens + num_token_per_tag)])
            num_existing_tokens += num_token_per_tag
        
        num_new_tokens = len(tags_to_update) * num_token_per_tag
    
    return train_dataset, eval_dataset, tag_name_dict, num_new_tokens, tags_to_update, domain_tags

 

