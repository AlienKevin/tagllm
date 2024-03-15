# Set up environment

```
conda env create -f environment-new.yml
conda activate tagllm-new
```

# Train
```
cd Tag-LLM
export KMP_DUPLICATE_LIB_OK=TRUE
python –m src.train
```