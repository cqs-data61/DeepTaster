# ACSAC23 Artifact for DeepTaster Submission

## Title
DeepTaster: Adversarial Perturbation-Based Fingerprinting to Identify Proprietary Dataset Use in Deep Neural Networks

## Overview
In this artifact, we provide the source code of DEEPTASTER, a DNN IP tracking tool that verifies whether an attacker’s model has been trained using a victim’s dataset or model, as well as pre-trained models. Pre-trained models can be used in testing as victim or suspect models which are used in Section 5 of our paper. The source code of DEEPTASTER consists of three parts: 1) DFT image generation (DFTimageGeneration.ipynb), 2) detection classifier training (DetectionClassifierGeneration.ipynb), and 3) evaluation suspect models (Evaluation.ipynb). Moreover, we provide the source code of three typical attack scenarios: transfer learning, fine-tuning, and pruning. To reconstruct DEEPTASTER, a CUDA-enabled GPU in Linux OS is required.

## Build Environment
we tested with the following versions of software:
1. Ubuntu 16.04
2. Python 3.7.10

## Prerequisite
Install foolbox [foolbox](https://github.com/bethgelab/foolbox), adversarial attack tool

Download tiny-imagenet

```python
$ pip install foolbox
$ chmod 755 download_tiny_imagenet.sh 
$ ./download_tiny_imagenet.sh /home_directory_path/TRACK-IP/
```
## Code File Organization

| File                         	| Functionality                                                       	|
| ---------------- | ------------------------------------------------------------ |
| train.py                    	        | Train models.                                                    	|
| DFTgeneration.py                      | Generate adversarial DFT images of a given model. 	|
| detection_classifier_generation.py  	| Generate detection_classifier using a set of adversarial DFT images. 	|
| evaluation.py                	        | Evaluate suspect models using DeepTaster. 	|
| deepsvdd.py                	          | Functions for DeepSVDD [1]. 	|
| requirements.txt                      | Python software requirements. 	|



[1] Ruff et al. "Deep One-Class Classification", Deep OCC ICML 2018

## To run 

### Generate Classifier

#### Step 1: Target model Generation
Training victim/suspect models need GPU and lots of time. You can freely use pre-trained models in [Google Drive](https://drive.google.com/drive/folders/1hWS5VssqjE0284YfL4mI9wJSTHyNsuN3).
You can download models.zip from the [Google Drive](https://drive.google.com/drive/folders/1hWS5VssqjE0284YfL4mI9wJSTHyNsuN3), unzip that file, and put them into the *models* folder or **can simply run *model.ipynb***.

#### Step 2: DFT images Generation

Run *DFTimageGeneration.ipynb*

#### Step 3: Detection classifier generation

Run *DetectionClassifierGeneration.ipynb*

Make sure "Cifar10" is the first located folder in the *images/val* folder.

### Evaluation Classifier

Run *DetectionClassifierGeneration.ipynb*

Make sure "Cifar10" is the first located folder in the *images/test* folder.

### Attack Model Generation


