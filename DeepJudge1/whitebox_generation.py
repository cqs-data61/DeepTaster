import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import time
from neuron_fuzz import NeuronFuzzing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare



parser = argparse.ArgumentParser(description='DeepJudge white-box test case generation process')
parser.add_argument('--model', required=True, type=str, help='victim model path')
parser.add_argument('--seeds', required=True, type=str, help='selected seeds path')
parser.add_argument('--layer', required=True, type=int, help='target layer index')
parser.add_argument('--m', default=3, type=float, help='hyper-parameter')
parser.add_argument('--iters', default=1000, type=int, help='iteration budget')
parser.add_argument('--neuron', type=int, help='target neuron index')
parser.add_argument('--dataset', type=str, help='training dataset')
parser.add_argument('--output', default='./testcases', type=str, help='test cases saved dir')


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


if __name__ == '__main__':
    TRAINING_NUMS = 20000
    NUMS = 100 
    
    opt = parser.parse_args()

    # load the victim model 
    #model_owner = load_model(opt.model)

    victim_model = torch.load(opt.model)
    ####cutting
    victim_model = nn.Sequential(*list(victim_model.children())[:opt.layer])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dummy_input = Variable(torch.randn(1, 3, 32, 32)).to(device)
    torch.onnx.export(victim_model, dummy_input, "mnist.onnx")
    model = onnx.load("mnist.onnx")
    tf_rep = prepare(model) 
    #tf_rep.export_graph('mnist.h5')

    with np.load(opt.seeds) as f:
        seeds_x = f['seeds_x']
    
    if opt.dataset == 'cifar10':
        cifar10 = tf.keras.datasets.cifar10
        (training_images, training_labels), (test_images, test_labels) = cifar10.load_data()
        x_train = training_images / 255.0
        x_train = x_train[:TRAINING_NUMS]
    elif opt.dataset == 'mnist':
        mnist = tf.keras.datasets.mnist
        (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
        training_images = training_images.reshape(60000, 28, 28, 1)
        x_train = training_images / 255.0
        x_train = x_train[:TRAINING_NUMS]
    else:
        x_train = None

    # generate test cases for a selected layer    
    fuzzer = NeuronFuzzing(tf_rep)
    start = time.time()
    tests = fuzzer.generate(seeds_x[:NUMS], layer_index=opt.layer, m=opt.m, iters=opt.iters, target_idx=opt.neuron, X=x_train)
    print("TIME COST", time.time()-start)
    
    log_dir = opt.output
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    save_path = f'{log_dir}/nf_layer{opt.layer}_ep{opt.m}.npy'
    np.save(save_path, tests)
    print('White-box test cases saved at ' + save_path)


