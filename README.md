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

#### Step 1: Target model Generation
Training victim/suspect models need GPU and lots of time. You can freely use pre-trained models in [URL](https://drive.google.com/drive/folders/1hWS5VssqjE0284YfL4mI9wJSTHyNsuN3).
You can download models from the [URL](https://drive.google.com/drive/folders/1hWS5VssqjE0284YfL4mI9wJSTHyNsuN3) and put them into the *models* folder or can simply run *model.ipynb*.

If you want to generate your own models, use the below commands.

```python
$ python train.py --dataset cifar10 --architecture Resnet18 --epochs 100
```

#### DFT images Generation
```python
$ python dftgeneration.py --model model_path --architecture Resnet18 --label 0 --type all --output save_image_directory
```

generate dft images


#### Detection classifier generation
```python
$ python detection_classifier_generation.py --dataset DFTdata_path --preepochs 40 --epochs 5 --output output_directory
```

#### Evaluation clssifier
```python
$ python evaluation.py --test test_DFTdata_path --classifier_dir classifier_directory
```


#### Attack

## Example
You can generate TRACK-IP for imagenet dataset protection using open source models by following below commands.

DFT image generation for victim models
Note that output_directory must have subdirectory temp, test, train, val
```python
$ pip install foolbox
$ python train.py --dataset cifar10 --architecture Resnet18
$ python train.py --dataset cifar10 --architecture Vgg16
$ python train.py --dataset cifar10 --architecture Densenet161
$ python train.py --dataset MNIST --architecture Resnet18
$ python train.py --dataset MNIST --architecture Vgg16
$ python train.py --dataset MNIST --architecture Densenet161
$ python dftgeneration.py --model ./cifar10/model/Resnet101_50.pt --architecture Resnet101 --label 0 --type all 
$ python dftgeneration.py --model ./cifar10/model/Vgg16_50.pt --architecture Vgg16 --label 0 --type all
$ python dftgeneration.py --model ./cifar10/model/Densenet161_50.pt --architecture Densenet161 --label 0 --type all
$ python dftgeneration.py --model ./MNIST/model/Resnet101_50.pt --architecture Resnet101 --label 1 --type test 
$ python dftgeneration.py --model ./MNIST/model/Vgg16_50.pt --architecture Vgg16 --label 1 --type test
$ python dftgeneration.py --model ./MNIST/model/Densenet161_50.pt --architecture Densenet161 --label 1 --type test
$ python detection_classifier_generation.py
$ python evaluation.py 
```
Detection classifier generation

