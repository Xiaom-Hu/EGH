# Embedding and Gradient Say Wrong: A White-Box Method for Hallucination Detection

The repo of paper [**Embedding and Gradient Say Wrong: A White-Box Method for Hallucination Detection**](https://aclanthology.org/2024.emnlp-main.116/)

## Run EGH in HaluEval

```bash
python EGH.py \
	--data_dir /path/to/HaluEval/data/qa_data.json \
	--task_type qa \
	--model_name /path/to/model \
	--train_ratio 0.1 \
	--lambda 0.2 \
```

## Citation

If this work is helpful, please kindly cite as:

```bibtex
@inproceedings{hu2024embedding,
  title={Embedding and Gradient Say Wrong: A White-Box Method for Hallucination Detection},
  author={Hu, Xiaomeng and Zhang, Yiming and Peng, Ru and Zhang, Haozhe and Wu, Chenwei and Chen, Gang and Zhao, Junbo},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={1950--1959},
  year={2024}
}
```
