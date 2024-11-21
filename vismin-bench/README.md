# VisMin Benchmark

## Overview

The VisMin Benchmark is designed to evaluate models on minimal-change tasks involving image-caption pairs. It supports two types of evaluation: Contrastive Models and Multimodal LLMs. The dataset consists of 2,084 samples with four types of minimal changes: object, attribute, count, and spatial relation.


## Dataset

Please download the dataset from [https://huggingface.co/datasets/mair-lab/vismin-bench].

## Evaluation Types

### 1. Contrastive Models

Following the dataset format, the contrastive task involves predicting the similarity between image-caption pairs:
The submission format is a csv file with five columns: `id`, `T0_I0`, `T0_I1`, `T1_I0`, `T1_I1`. 

```bash
id,T0_I0,T0_I1,T1_I0,T1_I1
```

Please compute similarity between text and image as following.

```python
T0_I0 = similarity(Text_0, Image_0)
T0_I1 = similarity(Text_0, Image_1)
T1_I0 = similarity(Text_1, Image_0)
T1_I1 = similarity(Text_1, Image_1)
```

The ground truth is available in the `solutions/sim_solution.csv` file.

### 2. Multimodal LLMs

This task involves visual question answering with four binary (yes/no) questions, four questions per sample. The submission format is a csv file with five columns: `id`, `Pred_Q0`, `Pred_Q1`, `Pred_Q2`, `Pred_Q3`.

```bash
id,Pred_Q0,Pred_Q1,Pred_Q2,Pred_Q3
```

You should predict the answer for each question based on the question and the image(s) provided in the dataset. The ground truth is available in the `solutions/vqa_solution.csv` file.

## How to evaluate your model

See an example of how to evaluate CLIP model in the `eval.py` file.

