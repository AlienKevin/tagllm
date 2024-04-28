# Setup Virtual Environment
```
conda create -n tagllm python=3.10 -y
conda activate tagllm
python -m pip install opencc-python-reimplemented datasets
```

# Dataset Generation
```
python gen_translation_dataset.py
```
