# ACSAC23 Artifact for DeepTaster Submission

## Title
DeepTaster: Adversarial Perturbation-Based Fingerprinting to Identify Proprietary Dataset Use in Deep Neural Networks

## Overview
DeepTaster is a DNN fingerprinting technique to track a victim's data unlawfully used to build a suspect model. DeepTaster generated adversarial perturbation images of victim models and transform them into Furier domain using DFT. By generating a classifier using DFT perturbations of victim models, DeepTaster can effectively identify data thefts. To show the effectiveness of DeepTaster, we prepare seven adversarial scenarios: MAA, DAA, SAA, TLA, MFA, MPA, and DATLA. 

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
You can download models from the [Google Drive](https://drive.google.com/drive/folders/1hWS5VssqjE0284YfL4mI9wJSTHyNsuN3) and put them into the *models* folder or **can simply run *model.ipynb***.

If you want to generate your own models, train nine models for each dataset and architecture using *train.ipynb*.

#### Step 2: DFT images Generation

Run *DFTimageGeneration.ipynb*

#### Step 3: Detection classifier generation

Run *DetectionClassifierGeneration.ipynb*

Make sure "Cifar10" is the first located folder in the *images/val* folder.

### Evaluation Classifier

Run *DetectionClassifierGeneration.ipynb*

Make sure "Cifar10" is the first located folder in the *images/test* folder.

### Attack Model Generation


