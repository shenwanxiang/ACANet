<a href="url"><img src="./doc/image/logo.png" align="left" height="130" width="140" ></a>



# CL-SAR
Contrastive learning of structure-activity relationship stduies (SAR)

Online triplet contrastive learning enables efficient cliff awareness in molecular activity prediction

------

## About

L_aca=L_mae+〖a*L〗_tsm

This study proposes the activity-cliff-awareness (ACA) loss for improving molecular activity prediction by deep learning models. The ACA loss enhances both metric learning in the latent space and task learning in the target space during training, making the network aware of the activity-cliff issue. For more details, please refer to the paper titled "Online triplet contrastive learning enables efficient cliff awareness in molecular activity prediction."

<p align="left" width="70%">
    <img width="100%" src="./doc/image/without_with_aca.png">
</p>

**Comparison of models for molecular activity prediction, one without (left) and one with (right) activity cliff awareness (ACA).**
The left panel depicts a model without ACA, where the presence of an activity cliff triplet (A, P, N) creates a challenge for the model to learn in the latent space. Due to the similarity between A and N in their chemical structures, the model learns graph representations that result in the distance between A-P being far greater than A-N, leading to poor training and prediction results. However, the right panel shows a model with ACA that optimizes the latent vectors in the latent space, making A closer to P and further away from N. The model with ACA combines metric learning in the latent space with minimizing the error for regression learning, while the model without ACA only focuses on the regression loss and may not effectively handle activity cliffs. 


## Performance

ACA loss vs. MAE loss on external test set and on No. of mined triplets during the training:
<p align="left" width="100%">
    <img width="40%" src="./doc/image/TestRMSE.png">
    <img width="40%" src="./doc/image/MinedTriplets.png">
</p>

More details on usage and performance can be found [here](https://github.com/bidd-group/bidd-clsar/blob/main/experiment/00_test/03_test_loss.ipynb).


## ACA loss implementation

* [Pytorch](https://github.com/bidd-group/bidd-clsar/blob/main/clsar/model/loss.py)
* [Tensorflow 2.x](https://github.com/bidd-group/bidd-clsar/blob/main/clsar/model/loss_tf.py)


## Installation


```bash
conda create -c conda-forge -n clsar rdkit
conda activate clsar
pip install -r ./requirements.txt 


```


