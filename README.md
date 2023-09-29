# ACSAC23 Artifact for DeepTaster Submission

## Title
DeepTaster: Adversarial Perturbation-Based Fingerprinting to Identify Proprietary Dataset Use in Deep Neural Networks

## Overview
This research artifact aims to provide the source code of DEEPTASTER, a DNN IP tracking tool that can identify unauthorized proprietary dataset use in deep neural networks. DEEPTASTER works by generating adversarial DFT images from a suspect model and then using a detection classifier to identify the dataset on which the suspect model was trained.

The source code of DEEPTASTER consists of three parts: 1) DFT image generation: This step generates adversarial DFT images from a suspect model, 2) Detection classifier training: This step trains a detection classifier to identify the dataset a suspect model was trained on, and 3) Evaluation of suspect models: This step evaluates the performance of DEEPTASTER by testing it on a variety of suspect models.

## Build Environment
We tested with the following versions of software:
1. Ubuntu 16.04
2. Python 3.7.10

## Prerequisite
To run DEEPTASTER, you will need the following:
1. Jupyter Notebook
2. Python 3.7
3. Other software listed in the requirements.txt file (include foolbox [foolbox](https://github.com/bethgelab/foolbox) )
4. A CUDA-enabled GPU in Linux OS

Using Cloud platform like **Google Colab** is an option for your reference.

## Prepare the environment


### Setting GPU
This repository require GPU environment. 

If you do not have a GPU environment configured, please refer to the instructions in [https://medium.com/geekculture/deep-learning-gpu-setup-from-scratch-75f730c49c01](https://medium.com/geekculture/deep-learning-gpu-setup-from-scratch-75f730c49c01) to configure one.

### Setting Anaconda
We recommend using a virtual environment. In the following steps, we will introduce how to set up a virtual environment using anaconda.

Step 1: install Anaconda
Implement anaconda following the instruction in [https://www.anaconda.com/](https://www.anaconda.com/) and [https://docs.anaconda.com/free/anaconda/install/linux/](https://docs.anaconda.com/free/anaconda/install/linux/).

Step 2: Create virtual environment
We provide anaconda.yml containing the elements needed to configure the environment.
To create the environment, run the following command.
```
  conda env create --file environment.yaml
```

Step 3: Activate environment
To activate environment, run the following command.
```
   conda activate DeepTaster_environment
```

### Setting Jupyter notebook
We recommend using Jupyter notebook to execute notebooks in the repository.

Installing Jupyter can be done simply by the following command.
```
  pip3 install jupyter
```

To run Jupyter notebook, execute the follosing command.
```
  jupyter notebook
```

### Download source codes
To download the repository, clone by the following command.
```
  git clone https://github.com/qkrtjsgp08/DeepTaster.git
```

## How To Run 

For quick implementation, we provide pre-trained attack models and DFT images. After installing the necessary packages in requirement.txt, you can run Simple_execution.ipynb to see all the results.

Alternatively, if you want to run it step-by-step, you can run it in the following order.

Note that you need to clairify the home directory path in the *DetectionClassifierGeneration.ipynb*, *Evaluation.ipynb*, *EvaluationAttackmodels.ipynb*, and *SImple_excution.ipynb*.

### Generate Classifier

#### Step 0: Target model generation
&emsp; Training victim/suspect models need GPU and lots of time. You can freely use pre-trained models in [Google Drive](https://drive.google.com/drive/folders/1Onxx5L77a16Vr3p10mvhWZ14VigqlkUm) by **simply running *download_models.ipynb***.

&emsp; Otherwise, you can download models.zip from the [Google Drive](https://drive.google.com/drive/folders/1Onxx5L77a16Vr3p10mvhWZ14VigqlkUm), unzip that file, and put them into the *models* folder.

#### Step 1: DFT images generation

&emsp; Run *DFTimageGeneration.ipynb*

#### Step 2: Detection classifier generation

&emsp; Run *DetectionClassifierGeneration.ipynb*

#### Step 3: Evaluation classifier

&emsp; Run *Evaluation.ipynb*



### Attack Model Generation

In this section, we provide source codes to generate seven different attak models: Fine-tuning, Transfer learning, Pruning, SAA (Same Architecrue Attack), DAA (Data Augmentation Attack), DATLA (Data Augmentation and Transfer Learning Attack), and TLPA (Transfer Learning with Pretrained model Attack).

#### Fine-tuning attack

&emsp; Run *FineTuning.ipynb*

&emsp; Fine-tune victim Resnet18 model and store attacked models at *models/attack_model/MFA*

#### Transfer learning attack

&emsp; Run *TransferLearning.ipynb*

&emsp; Transfer victim Resnet18 model on MNIST dataset and store attacked models at *models/attack_model/TLA*

#### Pruning attack

&emsp; Run *Pruning.ipynb*

&emsp; Prune victim Resnet18 model and store attacked models at *models/attack_model/MPA*

#### Same architecrue attack

&emsp; Run *SAA.ipynb*

&emsp; Prune victim Resnet18 model and store attacked models at *models/attack_model/SAA*

#### Data augmentation attack

&emsp; Run *DAA.ipynb*

&emsp; Prune victim Resnet18 model and store attacked models at *models/attack_model/DAA*

#### Data augmentation and transfer learning attack

&emsp; Run *DATLA.ipynb*

&emsp; Prune victim Resnet18 model and store attacked models at *models/attack_model/DATLA*

#### Transfer learning with pretrained model attack

&emsp; Run *TLPA.ipynb*

&emsp; Prune victim Resnet18 model and store attacked models at *models/attack_model/TLPA*

#### Evaluation attack models

&emsp; Run *EvaluationAttackmodels.ipynb*

&emsp; Evaluate all suspect models stored in *models/attack_model*

## Code File Organization

| File                         	| Functionality                                                       	|
| ---------------- | ------------------------------------------------------------ |
| requirements.txt                      | Python software requirements. 	|
| Simple_execution.ipynb                      | Evaluate DeepTaster using pretrained models and images	|
| download_models.ipynb                	          | Download pretrained classifier models. 	|
| DFTimageGeneration.ipynb               | Generate adversarial DFT images of models in *models* folder 	|
| DetectionClassifierGeneration.ipynb  	| Generate detection_classifier using a set of adversarial DFT images. 	|
| Evaluation.ipynb                	     | Evaluate suspect models using DeepTaster. 	|
| DAA.ipynb                	          | Generate data augmentation attack models. 	|
| SAA.ipynb                	          | Generate same architecture retraining attack models. 	|
| TransferLearning.ipynb                	          | Generate transfer learning attack models. 	|
| FineTuning.ipynb                	          | Generate fine-tuning attack models. 	|
| Pruning.ipynb                	          | Generate model pruning attack models. 	|
| DATLA.ipynb                	          | Generate data augmentation and transfer learing attack models. 	|
| TLPA.ipynb                	          | Generate transfer learning with pretrained model attack models. 	|
| EvaluationAttackmodels.ipynb                	          | Evaluate all attack models in *models/attack_model* forder. 	|
| ./utils/deepsvdd.py                	          | Functions for DeepSVDD [1]. 	|
| ./utils/train.py                	          | Train models. 	|

[1] Ruff et al., "Deep One-Class Classification," Deep OCC ICML 2018

## Resource usages

We are running the code on a server equipped with NVIDIA TITAN Xp GPU with CUDA v11.3. 
Runnig time reports the total time it took to execute the file.
GPU memory usage reports the maximum GPU memory usage.


| File                        	| Running time     	| GPU memory usage |
| ---------------- | ------------------------------------- | ------------------------------------- |
| Jupyter notebook                | -	| 3000MiB |
| Simple_execution.ipynb                | 30 minutes	| 9000MiB |
| download_models.ipynb                	| 10 minutes	| - |
| DFTimageGeneration.ipynb              | 50 minutes	| 2000MiB |
| DetectionClassifierGeneration.ipynb  	| 15 minutes 	| 700MiB |
| Evaluation.ipynb                	    | 10 minutes	| 800MiB |
| DAA.ipynb                	            | 3 hours	| 1000MiB |
| SAA.ipynb                	          | 1.5 hours 	| 1000MiB |
| TransferLearning.ipynb                  | 2.5 hours	| 9000MiB |
| FineTuning.ipynb                	          | 1 hour	| 7000MiB |
| Pruning.ipynb                	          | 1 hour 	| 9000MiB |
| DATLA.ipynb                	          | 2 hours 	| 1000MiB |
| TLPA.ipynb                	          | 2.5 hours	| 1000MiB |
| EvaluationAttackmodels.ipynb          | 1 hour	| 800MiB |

Different GPU might lead to quite different running time or GPU memory usage.
Note that the GPU memory usage depends on what dataset is used for training (using MNIST -> about 9000MiB, using CIFAR10 -> about 1000MiB).



