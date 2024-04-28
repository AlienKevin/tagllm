# Setup Virtual Environment
```
conda create -n tagllm python=3.10 -y
conda activate tagllm
python -m pip install opencc-python-reimplemented datasets evaluate sacrebleu scikit-learn
```

# Dataset Generation
The original datasets for translation are stored in `datasets/`. Run the following script to generate a new folder called `translations/` that collates all datasets together into a HuggingFace dataset:
```
python gen_translation_dataset.py
```

# Training
We adapt the [training script](https://huggingface.co/blog/mlabonne/orpo-llama-3) written by Maxime Labonne to fine-tune Llama 3 with QLoRA to our custom tasks and datasets. To speed up training and reduce memory requirement for the learnable tags configuration, we adapt Unsloth AI's hand-optimized Llama 3 [QLoRA fine-tuning script](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) to our tasks. See the `training_scripts/` folder for all the scripts used for each configuration and task in our paper:

| Configuration | Task | Script |
| -------- | ------- | -------- |
| No Tags  | Language Domain Adaptation | `qlora-lang-no-tag.ipynb` |
| No Tags  | POS Tagging | `qlora-pos-no-tag.ipynb` |
| No Tags  | Translation | `qlora-translation-no-tag.ipynb` |
| Natural Tags 1 | POS Tagging | `qlora-pos-no-lang.ipynb` |
| Natural Tags 1 | Translation | `qlora-translation-no-lang.ipynb` |
| Natural Tags 2 | Language Domain Adaptation | `qlora-lang.ipynb` |
| Natural Tags 2 | POS Tagging | `qlora-pos.ipynb` |
| Natural Tags 2 | Translation | `qlora-translation.ipynb` |
| Learnable Tags | Language Domain Adaptation | `tagllm-lang-unsloth.ipynb` |
| Learnable Tags | POS Tagging | `tagllm-pos-unsloth.ipynb` |
| Learnable Tags | Translation | `tagllm-translation-unsloth.ipynb` |

# Testing
We wrote our own test scripts for each configuration and task. Each test script should generate a JSONL output under the `experiment_results/` folder. See the `test_scripts/` folder for details:

| Configuration | Task | Script |
| -------- | ------- | -------- |
| Plain Llama 3 | POS Tagging | `test_pos.ipynb` |
| Plain Llama 3 | Translation | `test_translation.ipynb` |
| No Tags  | POS Tagging | `test_pos_qlora_no_tag.ipynb` |
| No Tags  | Translation | `test_translation_qlora_no_tag.ipynb` |
| Natural Tags 1 | POS Tagging | `test_pos_qlora_no_lang.ipynb` |
| Natural Tags 1 | Translation | `test_translation_qlora_no_lang.ipynb` |
| Natural Tags 2 | POS Tagging | `test_pos.ipynb` |
| Natural Tags 2 | Translation | `test_translation_qlora.ipynb` |
| Learnable Tags | POS Tagging | `test_pos_tagllm.ipynb` |
| Learnable Tags | Translation | `test_translation_tagllm.ipynb` |

# Evaluation
Run `python eval_pos.py` to generate `pos_evaluation_results.csv` under `experiment_results/`. We use HuggingFace's poseval module to reliably calculate the accuracy and F1 measures for the POS tagging task.

Similarly, run `python eval_translation.py` to generate `translation_evaluation_results.csv` under `experiment_results/`. We use the standard SacreBLEU package to reliably calculate the BLEU and chrF++ scores for the translation task.
