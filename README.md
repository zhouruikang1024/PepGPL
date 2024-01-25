# PepGPL
A deep learning model that combines graph neural networks and protein language models to simultaneously predict PepPIs and binding residues on both the peptide and protein sides.



# Requirements
* python == 3.7
* pytorch == 1.10.1
* Numpy == 1.21.5
* scikit-learn == 1.0.2
* dill == 0.36
* transformers == 4.20.1


# Files:

1.dataset

pep_prot_label_dataset_shuffle.csv: all peptide-protein pairs in the benchmark dataset.


2.Code

util_metric.py: computing evaluation metric. This function can evaluate the prediction performance of a classifier.

Network.py: This function contains the network framework of our entire model and is based on pytorch 1.10.1

train.py: This function is used for training model and testing the model performance under five-fold cross-validation.

predict.py: This function can rapidly test the predictive performance of our model by loading saved model parameters.

Example: 

Users can directly download our saved model parameters and preprocessed data features from the `./result` directory for model prediction.

```bash
python validation_model.py
```



# Train your own model PepGPL
Step 1, you need to use the SSPro, IUPRED2A and PSI-BLAST tools to compute the corresponding peptide and protein features.

Step 2, after generating the corresponding features, you can use the `data_preprocess.py` to preprocess the corresponding features to facilitate model input. 
The finally generated feature files are as follows:

peptide_feature_dict/protein_feature_dict: Encode the peptide and protein sequences according to the vocab.txt provided by ProtBERT.

peptide_mask_dict/protein_mask_dict: Save the true lengths of the peptide and protein sequences.

peptide_ss_dict/protein_ss_dict: Secondary structures of the peptide and protein sequences predicted by SSPro.

peptide_dense_dict/protein_dense_dict: The dense features for peptide sequences are three intrinsic disorder scores. The dense features for protein sequences are the concatenation of three intrinsic disorder scores and PSSM.

peptide_2_feature_dict/protein_2_feature_dict: Encode the physicochemical properties of each amino acid in the peptide and protein sequences.

Step3, Train own model PepGPL.

python model_train.py --path_dataset /Your path --epoch Your number --batch_size Your number

path_dataset: The path where the data is stored.

num_epochs: Define the maximum number of epochs.

batch_size: Define the number of batch size for training and testing.

Example:

```bash
cd train
python model_train.py --path_dataset ../dataset/pep_prot_label_dataset_shuffle.csv
```
# Contact 
If you have any questions or suggestions with the code, please let us know. Contact Ruikang Zhou at zhouruikang@csu.edu.cn
