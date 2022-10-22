import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from metrics import LOD, LAD, NOD, NAD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare
import torchvision.models as models

parser = argparse.ArgumentParser(description='DeepJudge white-box metric evaluation')
parser.add_argument('--model', required=True, type=str, help='victim model path')
parser.add_argument('--suspect', required=True, type=str, help='suspect model path or a dir')
parser.add_argument('--tests', required=True, type=str, help='test case saved path')
parser.add_argument('--output', default='./results', type=str, help='evaluation results saved dir')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

        
def log(content):
    if log_dir is not None:
        log_file = log_dir + '/whitebox_evaluation.txt'
        with open(log_file, 'a') as f:
            print(content, file=f)   
        


if __name__ == '__main__':
    opt = parser.parse_args()
    


    # load the victim model
    # victim_model = torch.load(opt.model)
    # victim_model.classifier = nn.Sequential(*list(victim_model.children())[:opt.layer])
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # dummy_input = Variable(torch.randn(1, 3, 32, 32)).to(device)
    # torch.onnx.export(victim_model, dummy_input, "mnist.onnx")
    # model = onnx.load("mnist.onnx")
    # model_owner= prepare(model)
    model_owner=torch.load(opt.model)

    # load the black-box test cases
    
    log_dir = opt.output
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    lod=0
    lad=0
    nod=0
    nad=0
    # metric evaluations
    if os.path.isfile(opt.suspect):
        # victim_model = torch.load(opt.suspect)
        # dummy_input = Variable(torch.randn(1, 3, 32, 32)).to(device)
        # torch.onnx.export(victim_model, dummy_input, "mnist.onnx")
        # model = onnx.load("mnist.onnx")
        # model_suspect= prepare(model)
        model_suspect=torch.load(opt.suspect)
        for root, dirs, files in os.walk(opt.tests):
          files.sort()
          file_num=len(files)
          for file in files:     
            tests =np.load(os.path.join(root,file), allow_pickle=True).item()
            lod += LOD(model_suspect, model_owner, tests) 
            lad += LAD(model_suspect, model_owner, tests) 
            nod += NOD(model_suspect, model_owner, tests) 
            nad += NAD(model_suspect, model_owner, tests) 
        print(f"victim model:{opt.model}, suspect model: {opt.suspect}")
        print(f"NOD: {nod/file_num}, LOD: {lod/file_num}, NAD: {nad/file_num}, LAD: {lad/file_num}")
        log(f"victim model:{opt.model}, suspect model: {opt.suspect}")
        log(f"NOD: {nod}, LOD: {lod}, NAD: {nad}, LAD: {lad}")
    elif os.path.isdir(opt.suspect):
        for root, dirs, files in os.walk(opt.suspect):
            files.sort()
            for file in files:
                model_suspect = load_model(os.path.join(root, file))
                #lod = LOD(model_suspect, model_owner, tests) 
                #lad = LAD(model_suspect, model_owner, tests) 
                nod = NOD(model_suspect, model_owner, tests) 
                #nad = NAD(model_suspect, model_owner, tests) 
                print(f"victim model:{opt.model}, suspect model: {file}")
                print(f"NOD: {nod}, LOD: {lod}, NAD: {nad}, LAD: {lad}")
                log(f"victim model:{opt.model}, suspect model: {file}")
                log(f"NOD: {nod}, LOD: {lod}, NAD: {nad}, LAD: {lad}")
                
     
    