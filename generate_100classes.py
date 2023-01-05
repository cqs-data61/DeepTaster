import pandas as pd
import os
DATA_DIR = './tiny-imagenet-200'
test_label=pd.read_csv(os.path.join(DATA_DIR,'val/val_annotations.txt'), sep="\t", header=None)
validation_len=len(test_label)
for i in range(validation_len):
    if test_label[1][i] in os.listdir(os.path.join(DATA_DIR,'val2')):
        os.system('cp ./tiny-imagenet-200/val/images/'+test_label[0][i]+' ./tiny-imagenet-200/val2/'+test_label[1][i]+"/images")