## Prerequisite

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


