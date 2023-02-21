## Prerequisite
Insatall foolbox, adversarial attack tool

Download tiny-imagenet

```python
$ pip install foolbox
$ chmod 755 download_tiny_imagenet.sh 
$ ./download_tiny_imagenet.sh /home_directory_path/TRACK-IP/
```
## Files
Files
attack: attack

Reference:

[1] Ruff et al. "Deep One-Class Classification", Deep OCC ICML 2018

## To run 

#### target model Generation
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
