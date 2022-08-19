
## To run 

#### Retraining attack
```python
$ python retrain_attack.py --dataset Imagenet --architecture Resnet101 --imagenetpath path_to_imagenet_dataset --output save_model_directory
```

#### 
```python
$ python 
```



#### Attack

## Example
You can generate TRACK-IP for imagenet dataset protection using open source models by following below commands.

DFT image generation for victim models
Note that output_directory must have subdirectory temp, test, train, val
```python
$ python retrain_attack.py --dataset Imagenet --architecture Resnet101 --imagenetpath path_to_imagenet_dataset --output save_model_directory
$ python retrain_attack.py --dataset Imagenet --architecture Vgg16 --imagenetpath path_to_imagenet_dataset --output save_model_directory
$ python retrain_attack.py --dataset Imagenet --architecture Densenet161 --imagenetpath path_to_imagenet_dataset --output save_model_directory
```
Detection classifier generation
