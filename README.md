# Set up environment

```
conda create -n tagllm python=3.10
conda activate tagllm
cd Tag-LLM
python -m pip install -r requirements.txt
```

# Train
```
cd Tag-LLM
export KMP_DUPLICATE_LIB_OK=TRUE
python –m src.train
```