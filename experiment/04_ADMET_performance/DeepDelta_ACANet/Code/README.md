## Code for Model Evaluation


#### models.py
* Functions for the following machine learning models:
  * [DeepDelta](https://github.com/RekerLab/DeepDelta)
  * [ChemProp](https://github.com/chemprop/chemprop) 
  * [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
  * [LightGBM](https://www.microsoft.com/en-us/research/project/lightgbm/) with both traditional and delta implementation
  * [ACANet](https://github.com/shenwanxiang/bidd-clsar/tree/main) 

#### cross_validations.py
* Test model performance using 5x10-fold cross-validation on 10 ADMET benchmarking datasets.

#### cross_validations_ACANetOPT.py
* Test model performance using 5x10-fold cross-validation on 10 ADMET benchmarking datasets by optimized ACANet

#### cross_validations_ACANetOOTB.py
* Test model performance using 5x10-fold cross-validation on 10 ADMET benchmarking datasets by out-of-the-box ACANet



#### external_test.py
* Test model performance on 2 ADMET external test sets. 

#### scaffold_chemical_similiarity_analysis.py
* Analyze the chemical similarity of pairs and compare the chemical pairs with shared scaffolds to pairs who do not share scaffolds. 

#### mathematical_invariants.py
* Calculate property differences of same molecule pairs for Eq 1 (with same molecule for both inputs, predictions should be zero): 
```math
DeepDelta(x,x)= 0. 
```

* Calculate reverse predictions for Eq. 2 (with swapped input molecules, predictions should be the inverse of the original predictions):
```math
DeepDelta(x,y)= -DeepDelta(y,x).
```

* Calculate predicted differences between three molecules for Eq. 3 (predicted difference between three molecules should be additive):
```math
DeepDelta(x,y) + DeepDelta(y,z)= DeepDelta(x,z)
```

