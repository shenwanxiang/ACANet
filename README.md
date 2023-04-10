<a href="url"><img src="./doc/image/logo.png" align="left" height="130" width="140" ></a>



# CL-SAR
Contrastive learning of structure-activity relationship stduies (SAR)

Online triplet contrastive learning enables efficient cliff awareness in molecular activity prediction

------

## About

This study proposes the activity-cliff-awareness (ACA) loss for improving molecular activity prediction by deep learning models. The ACA loss enhances both metric learning in the latent space and task learning in the target space during training, making the network aware of the activity-cliff issue. For more details, please refer to the paper titled "Online triplet contrastive learning enables efficient cliff awareness in molecular activity prediction."

## Performance

ACA loss vs. MAE loss on external test set and on No. of mined triplets during the training:

<p align="left" width="100%">
    <img width="40%" src="./doc/image/TestRMSE.png">
    <img width="40%" src="./doc/image/MinedTriplets.png">

</p>


## ACA loss 

* [pytorch](./doc/image/MinedTriplets.png)
* tensorflow



## Installation


```bash
conda create -c conda-forge -n clsar rdkit
conda activate clsar
pip install -r ./requirements.txt 


```


