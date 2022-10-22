import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare


parser = argparse.ArgumentParser(description='DeepJudge Seed Selection Process')
parser.add_argument('--model', required=True, type=str, help='victim model path')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset for the seed selection')
parser.add_argument('--num', default=1000, type=int, help='number of selected seeds')
parser.add_argument('--order', default='max', type=str, help='largest certainties or least. choice: max/min')
parser.add_argument('--output', default='./seeds', type=str, help='seeds saved dir')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def seedSelection(model, x, y, num=1000, order='max'):
    true_idx = np.where(np.argmax(model.run(x)[0], axis=1) == np.argmax(y, axis=1))[0]
    x, y = tf.gather(x,true_idx), tf.gather(y,true_idx)
    ginis = np.sum(np.square(model.run(x)[0]), axis=1)
    if order == 'max':
        ranks = np.argsort(-ginis)
    else:
        ranks = np.argsort(ginis)
    return tf.gather(x,ranks[:num]), tf.gather(y,ranks[:num])



if __name__ == '__main__':
    opt = parser.parse_args()
    
    if opt.dataset == 'cifar10':
        cifar100 = tf.keras.datasets.cifar10
        (training_images, training_labels), (test_images, test_labels) = cifar10.load_data()
        test_images =test_images.reshape(10000, 3, 32, 32)
    elif opt.dataset == 'mnist':
        mnist = tf.keras.datasets.mnist
        (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
        test_images = test_images.reshape(10000, 28, 28, 1)
    else:
        raise NotImplementedError()
    
    # select seeds from the testing dataset
    x_test = test_images / 255.0
    #x_test=x_test.astype(np.float32)
    y_test = tf.keras.utils.to_categorical(test_labels, 10)

    victim_model = torch.load(opt.model)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dummy_input = Variable(torch.randn(1, 3, 32, 32)).to(device)
    torch.onnx.export(victim_model, dummy_input, "mnist.onnx")
    model = onnx.load("mnist.onnx")
    tf_rep = prepare(model) 

    #cutting model to given layer

    #x_test_data=tf.expand_dims(x_test[0], axis=0)
    x_test_data=tf.cast(x_test, tf.float32)
    seeds_x, seeds_y = seedSelection(tf_rep, x_test_data, y_test, num=opt.num, order=opt.order)
    
    log_dir = opt.output
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    save_path = f"{log_dir}/{opt.dataset}_{opt.order}_{opt.num}seeds.npz"
    np.savez(save_path, seeds_x=seeds_x, seeds_y=seeds_y)
    print('Selected seeds saved at ' + save_path)


