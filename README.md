# CogQA

### [Project](https://sites.google.com/view/cognitivegraph/) | [arXiv](https://arxiv.org/abs/1905.05460)

**Cognitive Graph for Multi-Hop Reading Comprehension at Scale.**<br>
Ming Ding, Chang Zhou, Qibin Chen, Hongxia Yang and Jie Tang.<br>
In ACL 2019.

**Under construction…** 

The current version is a *preview* version, which means the codes are hard to read without comments. We **strongly recommend you to wait** until we tidy the codes up. 

We will release the final codes **as soon as possible**, and in principle will not answer any questions about the current preview version because there are actually some codes not easy to understand (or even insignificant bugs maybe QAQ). If you actually a quick follower on the same topic and still have some questions after carefully reading the codes, you can contact Ming Ding personally by email (See the [paper](https://arxiv.org/abs/1905.05460)). 

 ## Notes on  the preview version

* In preprocess, we refine the input into our *cognitive graph training samples* by fuzzy matching (`process_train.ipynb`). Then we save the fullwiki paragraphs in a redis database (`read_fullwiki.ipynb`).
* We train the model by running `run_cg.py` twice, once for mode *tensor* (task #1) and once for *bundle*(Task #2, load=True).
* Finally we can predict the answer by running `eval_cg.py`. 
* `hotpot_train_v1.1_500_refined.example.json` is an example input file and `hotpot_dev_fullwiki_v1_pred_with_cognitive_graph.json` is the result on dev set.
* Maybe `fullwiki_input_improved_by_cogqa1hop.zip` can directly improve your model on fullwiki setting by just replacing the original inputs~



If our work is useful to you, please cite our paper or star 🌟  our repo~~



