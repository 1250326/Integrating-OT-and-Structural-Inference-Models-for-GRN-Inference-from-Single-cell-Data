# Integrating Optimal Transport and Structural Inference Models for GRN Inference from Single-cell Data

This repository contains the official implementation of **Integrating Optimal Transport and Structural Inference Models for GRN Inference from Single-cell Data**.

## Setup
### Environments
Please note that the following is only a record of our environment, and we believe our code can adapt to a wide range of versions.
- Python: 3.10
- CUDA: 12.2
- numpy: 1.23.5
- pandas: 1.5.2
- PyTorch: 1.13.1
- scikit-learn: 1.2.0
- wot: 1.0.8.post2

### Data source
The simulated cellular trajectories of gene expression of different curated datasets can be generated by the BoolODE v0.1 ([link](https://github.com/Murali-group/BoolODE/releases/tag/v0.1)) with default networks.
Denote the file names of generated gene expression matrixes as `data/<net>_traj_combinded.npy`.
Following `data_generation_and_traj_reconstruction.ipynb`, you can get the train and test sets for mCAD and VSC we have used in our paper.

`data/<net>_traj_combinded.npy` is too big to be stored in GitHub, so we provided the intermediate `.h5ad` and `.csv` files under `/data`.
The ready-for-training VSC datasets are also provided under `/data`.

## Training
You can reproduce our results via:
```bash
save_folder="logs/"
net="VSC"
seed=0

python train_experiment.py --epochs 500 --batch-size 512 --suffix "scRNAseq" --save-folder $save_folder --edge-types 2 --prediction-steps 1 --prior --save-probs --file-name "data/${net}_traj_reconstructed_7t_test.npy" --smoothness-weight=-500 --exp-name "${net}_exp${seed}" --encoder gin --decoder mlp --encoder-dropout 0.3 --decoder-dropout 0.5 --seed $seed &
```

## Evaluation
We use three metrics for evaluation:
- AUROC: This is calculated with `sklearn.metrics.roc_auc_score`
- AUPR: This is calculated with `sklearn.metrics.average_precision_score`
- EPR: We follow the definition of EPR in BEELINE and calculated as follow:
```python
def cal_epr(ref_net, preds):
    return len(set(np.argpartition(preds.reshape(-1), -ref_net.sum())[-ref_net.sum():]).intersection(set(np.where(ref_net.reshape(-1))[0]))) / ref_net.mean() / ref_net.sum()
```

<!-- ## Citation
To cite our work, please use the following:
```
``` -->