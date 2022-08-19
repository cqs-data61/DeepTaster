
## To run 

#### Retraining attack
Generate 7 models each of them is trained on patial dataset (10%, 30%, 50%, 70%, 80%, 90%, and 100% of total dataset)
```python
$ python retrain_attack.py --dataset Imagenet --architecture Resnet101 --imagenetpath path_to_imagenet_dataset --output save_model_directory
```
#### Fine-tuning attack
Generate 4 models each of them is fine tuning model trained on patial dataset (100, 500, 1000, 2000 samples of target dataset)

CIFAR100 dataset cases
```python
$ python fine_tuning.py --model victim_model_path --dataset cifar100 --output save_model_directory
```
IMAGENET dataset cases
```python
$ python fine_tuning.py --dataset Imagenet --architecture Resnet101 --imagenetpath path_to_imagenet_dataset --output save_model_directory
```


#### Attack

## Example
Retraining attack
```python
$ python retrain_attack.py --dataset Imagenet --architecture Resnet101 --imagenetpath path_to_imagenet_dataset --output save_model_directory
$ python retrain_attack.py --dataset Imagenet --architecture Vgg16 --imagenetpath path_to_imagenet_dataset --output save_model_directory
$ python retrain_attack.py --dataset Imagenet --architecture Densenet161 --imagenetpath path_to_imagenet_dataset --output save_model_directory
```
Finetuning attack
```python
$ python
```
