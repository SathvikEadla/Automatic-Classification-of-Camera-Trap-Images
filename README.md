# Requirements:
	- python > 3.5
	- pandas
	- numpy
	- sklearn
	- matplotlib
	- Tensorflow GPU
	
# Installation:

```sh
$ pip install libmr, pandas, numpy, sklearn, matplotlib, seaborn, hyperopt
```	

### Installing Tensorflow-GPU
The best method to install tensorflow-gpu in your conda environment is to use ***conda install -c anaconda tensorflow-gpu***.
	
```sh
$ conda install -c anaconda tensorflow-gpu
```

# Usage:

	- "train_model.py" :
		Used to train modified pre-trained models with the desired dataset. 
		The Dataset should contain ***class_names*** for *sub-directories* inside the main directory.
		The dataset using in this work is sample of ***Snapshot Serengeti Dataset***.
		
	- "Serengeti_GUI.py" 
		contains code for loading the trained models and to initialize GPU for feeding images to model and an output window to view results.
		This GUI is designed for using with wildlife data.
		
# Screenshots:
These are the screenshots of GUI designed for **Automatic Classification of Camera Trap Images**

![Screenshot of GUI](/Screenshots/Picture1.png)

![Screenshot of GUI](/Screenshots/Picture2.png)	

# For Using Snapshot Serengeti Dataset:

The dataset can be downloaded from [here](http://lila.science/datasets/snapshot-serengeti)

This data set contains approximately 2.65M sequences of camera trap images, totaling 7.1M images, from seasons one through eleven of the Snapshot Serengeti project, the flagship project of the Snapshot Safari network. Using the same camera trapping protocols at every site, Snapshot Safari members are collecting standardized data from many protected areas in Africa, which allows for cross-site comparisons to assess the efficacy of conservation and restoration programs. 

Labels are provided for 61 categories, primarily at the species level (for example, the most common labels are wildebeest, zebra, and Thomson’s gazelle). Approximately 76% of images are labeled as empty. A full list of species and associated image counts is available [here](https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/SnapshotSerengeti_S1-11_v2.1.species_list.csv). 


# Attribution:

The images and species-level labels are described in more detail in the associated manuscript:

Swanson AB, Kosmala M, Lintott CJ, Simpson RJ, Smith A, Packer C (2015) Snapshot Serengeti, high-frequency annotated camera trap images of 40 mammalian species in an African savanna. Scientific Data 2: 150026. 
