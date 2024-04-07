import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
import os
PTQ4ViT_PATH = f"./submodules/PTQ4ViT"
sys.path.append(PTQ4ViT_PATH) # Path to PTQ4ViT project

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from importlib import reload,import_module
import multiprocessing
import os
import time
from itertools import product

import utils.datasets as datasets
import utils.net_wrap as net_wrap
from utils.quant_calib import QuantCalibrator, HessianQuantCalibrator
from utils.models import get_net

import numpy as np
import random

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 使用你的随机种子
# set_seed(3)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='vit_base_patch16_384')
    parser.add_argument("--datapath", type=str, default='/open_datasets/imagenet/')
    parser.add_argument("--quantizer", type=str, default='PTQ4ViT',
                        help="quantizer name", choices=["BasePTQ","PTQ4ViT"])
    
    parser.add_argument("--wbits", type=int, default=6)
    parser.add_argument("--abits", type=int, default=6)
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--n-bc-samples", type=int, default=32)

    parser.add_argument("--n_gpu", type=int, default=6)
    parser.add_argument("--multiprocess", action='store_true')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument('--use-bc', action='store_true',help='Whether to apply the bias compensation in model quantization')
    parser.add_argument('--debug-bc', action='store_true',help='Whether to show the bias compensation error reduce')
    parser.add_argument('--layer-wise', action='store_true',help='Whether to apply the bias compensation in model quantization')
    parser.add_argument('--align-attn', action='store_true',help='Whether to apply the bias compensation for attn blocks')
    parser.add_argument('--align-mlp', action='store_true',help='Whether to apply the bias compensation for mlp blocks')
    parser.add_argument('--align-blocks', action='store_true',help='Whether to apply the bias compensation for model blocks')
    parser.add_argument('--n-align-layer', action='store_true',help='Do not apply the bias compensation to quantized layers')
    parser.add_argument('--n-align-model', action='store_true',help='Do not apply the bias compensation to quantized model')
    parser.add_argument('--gsfb', action='store_true',help='grid-search-for-best bc strategy. Set this will invalid the align settings and auto search a best align strategies')
    parser.add_argument("--quantize-pos", type=int, default=7, 
                        help='quantize position: 001 enable bc for Conv Layer, 010 enable bc for FC Layer, 100 enable bc for BMM Layer, 111 for all layers')
    parser.add_argument('--test-pose', action='store_true',help='test bc position for different vits. Set this will invalid the quantize-pos arguments')
    parser.add_argument("--save-path", type=str, default='./models')
    parser.add_argument("--save-quantized-model", action='store_true', help = 'save the quantized model to the save path')
    parser.add_argument("--load-quantized-model", action='store_true', help = 'load the quantized model from the save path')
    parser.add_argument("--test-quantized-model", action='store_true', help = 'test the quantized model accuracy')


    # logging
    parser.add_argument('--log-root', default='./logs', type=str, help='path to save logs', required=False)
    parser.add_argument('--verbose', action='store_true', help='print log to screen')
    
    parser.add_argument(
        '--seed',
        type=int, default=3, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--bc-seed',
        type=int, default=3, help='Seed for sampling the calibration data for BC.'
    )

    args = parser.parse_args()
    return args

def test_classification(net,test_loader,max_iteration=None, description=None):
    pos=0
    tot=0
    i = 0
    max_iteration = len(test_loader) if max_iteration is None else max_iteration
    with torch.no_grad():
        q=tqdm(test_loader, desc=description)
        for inp,target in q:
            i+=1
            inp=inp.cuda()
            target=target.cuda()
            out=net(inp)
            pos_num=torch.sum(out.argmax(1)==target).item()
            pos+=pos_num
            tot+=inp.size(0)
            q.set_postfix({"acc":pos/tot})
            if i >= max_iteration:
                break
    print(pos/tot)
    return pos/tot

def process(pid, experiment_process, args_queue, n_gpu):
    """
    worker process. 
    """
    gpu_id=pid%n_gpu
    os.environ['CUDA_VISIBLE_DEVICES']=f'{gpu_id}'

    tot_run=0
    while args_queue.qsize():
        test_args=args_queue.get()
        print(f"Run {test_args} on pid={pid} gpu_id={gpu_id}")
        experiment_process(**test_args)
        time.sleep(0.5)
        tot_run+=1
        # run_experiment(**args)
    print(f"{pid} tot_run {tot_run}")


def multiprocess(experiment_process, cfg_list=None, n_gpu=6):
    """
    run experiment processes on "n_gpu" cards via "n_gpu" worker process.
    "cfg_list" arranges kwargs for each test point, and worker process will fetch kwargs and carry out an experiment.
    """
    args_queue = multiprocessing.Queue()
    for cfg in cfg_list:
        args_queue.put(cfg)

    ps=[]
    for pid in range(n_gpu):
        p=multiprocessing.Process(target=process,args=(pid,experiment_process,args_queue,n_gpu))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()

def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _,_,files =  next(os.walk(f"{PTQ4ViT_PATH}/configs"))
    if config_name+".py" in files:
        quant_cfg = import_module(f"configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg
        

def experiment_basic(net='vit_base_patch16_384', config="PTQ4ViT"):
    """
    A basic testbench.
    """
    quant_cfg = init_config(config)
    net = get_net(net)
    wrapped_modules = net_wrap.wrap_modules_in_net(net,quant_cfg)
    
    g=datasets.ViTImageNetLoaderGenerator('/datasets/imagenet','imagenet',32,32,16,kwargs={"model":net})
    test_loader=g.test_loader()
    calib_loader=g.calib_loader(num=32)
    
    quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=4) # 16 is too big for ViT-L-16
    quant_calibrator.batching_quant_calib()
    
    test_classification(net,test_loader)

if __name__=='__main__':
    args = parse_args()
    cfg_list = []

    nets = ['vit_tiny_patch16_224', "deit_base_patch16_384"]
    configs= ['PTQ4ViT']

    cfg_list = [{
        "net":net,
        "config":config,
        }
        for net, config in product(nets, configs) 
    ]

    if args.multiprocess:
        multiprocess(experiment_basic, cfg_list, n_gpu=args.n_gpu)
    else:
        for cfg in cfg_list:
            experiment_basic(**cfg)
    

