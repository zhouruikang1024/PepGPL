# PepGPL
A deep learning model that combines graph neural networks and protein language models to simultaneously predict PepPIs and corresponding peptide-binding residues.



# Requirements
* python == 3.7
* pytorch == 1.10.1
* Numpy == 1.21.5
* scikit-learn == 1.0.2
* dill == 0.36


# Files:

1.dataset

positive_negative_pair.csv: all peptide-protein pairs in the benchmark dataset.


2.Code

util_metric.py: computing evaluation metric. This function can evaluate the prediction performance of a classifier.

Network.py: This function contains the network framework of our entire model and is based on pytorch 1.10.1

model_train.py: This function is used for retraining model and testing the model performance under five-fold cross-validation..

validation_model.py: This function can rapidly test the predictive performance of our model by loading saved model parameters.

Example:

```bash
python validation_model.py
```
