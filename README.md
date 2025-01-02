# EGH

The repo of paper [**Embedding and Gradient Say Wrong: A White-Box Method for Hallucination Detection**](https://aclanthology.org/2024.emnlp-main.116/)

## Run EGH in HaluEval

```bash
python EGH.py \
	--data_dir /path/to/HaluEval/data/qa_data.json \
	--task_type qa \
	--model_name /path/to/model \
	--train_ratio 0.1 \
    --lambda 0.2 
```

