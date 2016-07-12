# cnn-bcdr

Source code for [Representation learning for mammography mass lesion classification with convolutional neural networks](http://www.sciencedirect.com/science/article/pii/S0169260715300110) ([pdf](https://www.researchgate.net/profile/John_Arevalo/publication/289585420_Representation_learning_for_mammography_mass_lesion_classification_with_convolutional_neural_networks/links/570ec9bc08aee328dd654afe.pdf)). 

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7
* Pylearn2 (tested on [pylearn2@30437ee](https://github.com/lisa-lab/pylearn2/tree/30437ee))
* Scipy
* [Shapely] (https://github.com/Toblerity/Shapely)

## Usage

### Data

With this paper we released the Breast Cancer Digital Repository F03 (BCDR-F03) dataset. You can get a copy from http://bcdr.inegi.up.pt/. Uncompress it under the `data` folder.

### Preprocessing
  * Create hdf5 dataset:

    ```
    python make_dataset.py config.yaml
    ```
  * Build preprocessed version (GCN + LCN):

    ```
    python preprocessing.py config.yaml
    ```
### Training
The hyperparameters to train the network are in the `config.yaml` file. Train the model:

```
python train_cnn.py config.yaml
```

### Evaluation
Evaluate trained model:

```
python eval.py config.yaml
```
