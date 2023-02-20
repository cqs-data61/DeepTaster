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
$ python train.py --dataset cifar10 --architecture Resnet18
```

#### DFT images Generation
```python
$ python dftgeneration.py --model model_path --type all --output save_image_directory
```

generate dft images


#### Detection classifier generation
```python
$ python detection_classifier+generation.py --train train_data_path --val validataion_data_path --saveautoencoder save_autoencoder_directory --output save_classifier_directory
```



#### Attack

## Example
You can generate TRACK-IP for imagenet dataset protection using open source models by following below commands.

DFT image generation for victim models
Note that output_directory must have subdirectory temp, test, train, val
```python
$ pip install foolbox
$ python dftgeneration.py --model Imagenet --architecture Resnet101 --type all --output output_directory
$ python dftgeneration.py --model Imagenet --architecture Vgg16 --type all --output output_directory
$ python dftgeneration.py --model Imagenet --architecture Densenet161 --type all --output output_directory
$ python detection_classifier_generation.py --train output_directory/train/ --val output_directory/val/ --output classifier_saved_directory
```
Detection classifier generation
