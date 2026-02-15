# Weak Models Can Be Good Teachers: A Case Study on Link Prediction with MLPs

This repository contains the official implementation of:

> **Weak Models Can Be Good Teachers: A Case Study on Link Prediction with MLPs**  
> Learning on Graphs (LoG) 2025

OpenReview: https://openreview.net/forum?id=TSwEOuoO00  

---


## Environment

Please install environments for https://github.com/Juanhui28/HeaRT and https://github.com/snap-research/linkless-link-prediction 

# 1. Generate Edges Data for Distillation

See linkless-link-prediction/src/get\_distill\_edges.py, this is modified from linkless link prediction's main.py

```bash
python get_distill_edges.py --datasets=cora
```
The edge file is stored in linkless-link-prediction/data/cora\_edges.pth

# 2. Generate Teacher Guidance

We modify the code in https://github.com/Juanhui28/HeaRT to generate guidance from teachers. You can follow the examples below to generate guidance for any models implemented in HeaRT.

HeaRT/benchmarking/exist\_setting\_ogb/main\_heuristic\_ogb.py

HeaRT/benchmarking/exist\_setting\_ogb/main\_gnn\_ogb.py

HeaRT/benchmarking/exist\_setting\_small/main\_gnn\_CoraCiteseerPubmed.py

HeaRT/benchmarking/exist\_setting\_small/main\_heuristic\_CoraCiteseerPubmed.py

Example command

```bash
python main_heuristic_CoraCiteseerPubmed.py --data_name cora --edge_path_file ../../../linkless-link-prediction/data/cora_edges.pth --use_heuristic CN

python main_heuristic_CoraCiteseerPubmed.py --data_name cora --edge_path_file ../../../linkless-link-prediction/data/cora_edges.pth --use_heuristic AA
```

The teacher guidance is saved in HeaRT/benchmarking/exist\_setting\_small/cora\_CN_\pred.pth and HeaRT/benchmarking/exist\_setting\_small/cora\_AA\_pred.pth


# 3. Train Student from Single Teacher

The student training is based on the code in https://github.com/snap-research/linkless-link-prediction


linkless-link-prediction/src/get\_heuristic.py

linkless-link-prediction/src/heart\_distill\_grid\_search.py

Example command

```bash
python heart_distill.py --distill_teacher CN --datasets cora --distill_pred_path ../../HeaRT/benchmarking/exist_setting_small/cora_CN_pred.pth --lr 0.01 --dropout 0.5 --LLP_D 0.01 --LLP_R 1 --True_label 1

python heart_distill.py --distill_teacher AA --datasets cora --distill_pred_path ../../HeaRT/benchmarking/exist_setting_small/cora_AA_pred.pth --lr 0.01 --dropout 0.5 --LLP_D 0.01 --LLP_R 1 --True_label 1
```
# 4. Train Ensemble MLPs

linkless-link-prediction/src/ensemble\_distilled\_mlp.py

Example command
```bash
python ensemble_distilled_mlp.py --datasets cora
```
