## Prerequisite
We use foolbox, adversarial attack tool

```python
$ pip install foolbox
'''
## Files
Files
attack: attack

Reference:

[1] Ruff et al. "Deep One-Class Classification", Deep OCC ICML 2018

## To run 

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
```python
$ pip install foolbox
$ python dftgeneration.py --model Imagenet --architecture Resnet101 --type all --output output_directory1
$ python dftgeneration.py --model Imagenet --architecture Vgg16 --type all --output output_directory2
$ python dftgeneration.py --model Imagenet --architecture Densenet161 --type all --output output_directory3
```
