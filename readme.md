# CogQA

### [Project](https://sites.google.com/view/cognitivegraph/) | [arXiv](https://arxiv.org/abs/1905.05460)

Source codes for the paper **Cognitive Graph for Multi-Hop Reading Comprehension at Scale.**  *(ACL 2019 Oral)* 

We also have a [Chinese blog](https://zhuanlan.zhihu.com/p/72981392) about CogQA on Zhihu (知乎) besides the [paper](https://arxiv.org/abs/1905.05460).

## Introduction

CogQA is a novel framework for multi-hop question answering in **web-scale** documents. Founded on the dual process theory in cognitive science, CogQA gradually builds a *cognitive graph* in an iterative process by coordinating an implicit extraction module (System 1) and an explicit reasoning module (System 2). While giving accurate answers, our framework further provides **explainable** reasoning paths. 

## Preprocess

1. Download and setup Redis database following https://redis.io/download
2. Download the dataset, evalute script and fullwiki data (enwiki-20171001-pages-meta-current-withlinks-abstracts) from https://hotpotqa.github.io. Unzip `improved_retrieval.zip` in this repo.
3. ``pip install -r requirements.txt``
4. Run ``python read_fullwiki.py`` to load wikipedia documents to redis (check the size of `dump.pkl` is about 2.4GB).
5. Run ``python process_train.py`` to generate `hotpot_train_v1.1_refined.json`, which contains edges in gold-only cognitive graphs.
6. ``mkdir models``

## Training

The codes automatic assign tasks on all available devices, each handling `batch_size / num_gpu` samples. We recommend that each gpu has at least 11GB memory to hold 2 batch.

1. Run `python train.py` to train Task #1(span extraction).
2. Run `python train.py --load=True --mode='bundle'` to train Task #2(answer prediction).

## Evaluation

The `cogqa.py` is the algorithm to answer questions with a trained model. We split the 1-hop nodes found by another similar model into `improved_retrieval.zip` for reuse in other algorithm. It  can **directly** improve your result on fullwiki setting by just replacing the original input.

1. unzip  ` improved_retrieval.zip`.

2. `python cogqa.py --data_file='hotpot_dev_fullwiki_v1_merge.json'`
3. `python hotpot_evaluate_v1.py hotpot_dev_fullwiki_v1_merge_pred.json hotpot_dev_fullwiki_v1_merge.json` 
4. You can check the cognitive graph (reasoning process) in the `cg` part of the predicted json file.

## Notes

1. The changes of this version from the preview version is mainly about **detailed comments**.
2. The relatively sensetive hyperparameters includes the number of  negative samples, top K, learning rate of task #2, scale factors between different parts...
3. If our work is useful to you, please cite our paper or star 🌟  our repo~~
