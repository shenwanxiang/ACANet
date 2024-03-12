# CL-SAR
<a href="url"><img src="./doc/image/logo.png" align="left" height="130" width="140" ></a>
[![Dataset](https://img.shields.io/badge/dataset-MPCD-green)](https://github.com/bidd-group/MPCD)
[![Paper](https://img.shields.io/badge/paper-Researchsquare-yellow)](https://www.researchsquare.com/article/rs-2988283/v1)
[![Paper](https://img.shields.io/badge/paper-Chemrxiv-orange)](https://chemrxiv.org/engage/chemrxiv/article-details/6470c963be16ad5c57f5526c)
[![Codeocean](https://img.shields.io/badge/reproduction-Codeocean-9cf)](https://codeocean.com/capsule/8102819/tree/v1)
[![PyPI version](https://badge.fury.io/py/clsar.svg)](https://badge.fury.io/py/clsar)
[![Downloads](https://static.pepy.tech/badge/clsar)](https://pepy.tech/project/clsar)

----------
CLSAR: Contrastive learning of structure-activity relationship stduies (SAR)


----------
## About

Online triplet contrastive learning enables efficient cliff awareness in molecular activity prediction

```math
$$\mathcal{L}_{aca}=\mathcal{L}_{mae}+{a\ast\mathcal{L}}_{tsm}$$
```


This study proposes the activity-cliff-awareness (ACA) loss for improving molecular activity prediction by deep learning models. The ACA loss enhances both metric learning in the latent space and task learning in the target space during training, making the network aware of the activity-cliff issue. For more details, please refer to the paper titled "Online triplet contrastive learning enables efficient cliff awareness in molecular activity prediction."

<p align="left" width="65%">
    <img width="100%" src="./doc/image/without_with_aca.png">
</p>

**Comparison of models for molecular activity prediction, one without (left) and one with (right) activity cliff awareness (ACA).**
The left panel depicts a model without ACA, where the presence of an activity cliff triplet (A, P, N) creates a challenge for the model to learn in the latent space. Due to the similarity between A and N in their chemical structures, the model learns graph representations that result in the distance between A-P being far greater than A-N, leading to poor training and prediction results. However, the right panel shows a model with ACA that optimizes the latent vectors in the latent space, making A closer to P and further away from N. The model with ACA combines metric learning in the latent space with minimizing the error for regression learning, while the model without ACA only focuses on the regression loss and may not effectively handle activity cliffs. 


## Performance

ACA loss vs. MAE loss on external test set and on No. of mined triplets during the training:
<p align="left" width="100%">
    <img width="45%" src="./doc/image/TestRMSE.png">
    <img width="48%" src="./doc/image/MinedTriplets.png">
</p>

More details on usage and performance can be found [here](https://github.com/bidd-group/bidd-clsar/blob/main/experiment/00_test/04_test_loss_HSSMS.ipynb).



## ACA loss implementation

* [Pytorch](https://github.com/bidd-group/bidd-clsar/blob/main/clsar/model/loss.py)
* [Tensorflow 2.x](https://github.com/bidd-group/bidd-clsar/blob/main/clsar/model/loss_tf.py)



## ACA loss usage 
```python

#Pytorch
from clsar.model.loss import ACALoss
aca_loss = ACALoss(alpha=0.1, cliff_lower = 0.2, cliff_upper = 1.0, p = 1., squared = False)
loss = aca_loss(labels,  predictions, embeddings)
loss.backward()


#Tensorflow
from clsar.model.loss_tf import ACALoss

```


## Installation

```bash
pip install clsar
```




## Run ACANet

```python
from clsar import ACANet
#Xs_train: list of SMILES string of training set
#y_train_pIC50: the pChEMBL labels of training set

## init ACANet
clf = ACANet(gpuid = 0,   work_dir = './')

## get loss hyperparameters (cliff_lower, cliff_upper, and alpha) by training set 
dfp = clf.opt_cliff_by_cv(Xs_train, y_train_pIC50, total_epochs=50, n_repeats=3)
dfa = clf.opt_alpha_by_cv(Xs_train, y_train_pIC50, total_epochs=100, n_repeats=3)


## fit model using 5fold cross-validation
clf.cv_fit(Xs_train, y_train_pIC50, verbose=1)


## make prediction using the 5-submodels, the outputs are the average of the 5-submodels
test_pred_pIC50 = clf.cv_predict(Xs_test)
```


## Citation

Wan Xiang Shen*, Chao Cui*, Jian Ming Wang*, Xiang Cheng Shi et al. `Online triplet contrastive learning enables efficient cliff awareness in molecular activity prediction`, 28 June 2023, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-2988283/v1].



