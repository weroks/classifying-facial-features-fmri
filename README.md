# classifying-facial-features-fmri
Code related to my bachelor thesis on "Classifying Facial Features from fMRI Patterns".

### This repository contains:
* code and data related to the paper: ["Reconstructing Faces from fMRI Patterns using Deep Generative Neural Networks"](https://www.nature.com/articles/s42003-019-0438-y) by VanRullen &amp; Reddy (2019):
	* the files make_network.py, utils.py and vaegan_celeba.ckpt to load the pretrained VAE-GAN
	* the folder "derivatives", which contains files necessary for brain decoding
	* this and more code can also be accessed via their own [GitHub repository](https://github.com/rufinv/VAE-GAN-CelebA) 
* code to classify the latent vectors decoded by VanRullen &amp; Reddy (2019) ("clf_latent_vecs.py) and some arrays containing the training vectors ("vecs") and labels ("true_exp")
* code to create/format the preprocessed fMRI data ("create_fmri_dataset.py")
	* this code was used to create the numpy arrays "train_samples", "train_targets", "test_samples" and "test_targets", which are also available
	* to execute this code a folder with the preprocessed data is needed, which was too big to upload it here, but can be made available upon request
* code to classify the fMRI dataset (clf_fmri.py)
* the following files were to big to upload them here and can therefore be accessed via the link below:
	* veagan_celeba.ckpt
	* derivatives
	* train_samples.npy
* [the link](https://mega.nz/file/ogpyQAYB#5mILfvDdaKYkyybrVu2pkKLWtI7j9iEXEsO7oaJvw60)

### Requirements:
* Python >= 3.4
* tensorflow, numpy, scipy, sklearn, pandas, h5py, pathlib, mvpa2
