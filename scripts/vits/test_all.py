# -*- coding: utf-8 -*-
import os
# os.environ['http_proxy'] = 'http://127.0.0.1:4780'
# os.environ['https_proxy'] = 'http://127.0.0.1:4780'
data_cache_dir = "./cache_dir"

# from timm.models.layers import config
from torch.nn.modules import module
from test_vit_utils import *

import sys
sys.path.append(PTQ4ViT_PATH) # Path to PTQ4ViT project

from quant_layers.conv import MinMaxQuantConv2d, PTQSLQuantConv2d
from quant_layers.linear import MinMaxQuantLinear, PTQSLQuantLinear
from quant_layers.matmul import MinMaxQuantMatMul, PTQSLQuantMatMul
import matplotlib.pyplot as plt
from utils.net_wrap import wrap_certain_modules_in_net
from tqdm import tqdm
import torch.nn.functional as F
import pickle as pkl
from itertools import product
import types
from utils.quant_calib import HessianQuantCalibrator, QuantCalibrator
from utils.models import get_net
import time

import logging
from pprint import pprint
import copy

formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# 创建一个日志记录器
logger = logging.getLogger('training')
logger.setLevel(logging.DEBUG)
# 创建一个文件处理器，并Formatter 添加到处理器
# file_handler = logging.FileHandler('experiments.log')
# file_handler.setFormatter(formatter)
# 创建一个控制台处理器，并将 Formatter 添加到处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
# 将处理器添加到记录器
# logger.addHandler(file_handler)
logger.addHandler(console_handler)

@torch.no_grad()
def apply_bias_compensation(float_model, model, dataloader, dev,verbose=False):
    DEBUG_BC = args.debug_bc
    import sys
    from bias_compensation.quantize.quantize import quantize_module_weight, quantize_module_act,quantize_model
    from bias_compensation.quantizers.ul2ptq import UL2PTQ, UL2PTQ2, UL2PTQ3
    from bias_compensation.utils.general_utils import (
        freeze_all_modules,
        enable_training_modules,
        stat_trainable_params,
        GetInputsOutputs
    )
    import numpy as np
    import copy
    if verbose:
        logger.info('Starting BC ...')

    def find_layers(t_model,allow_list=()):
        subsets = []
        for name, module in t_model.named_modules():
            if isinstance(module, allow_list):
                subsets.append(module)
        return subsets

    # find all the quantizing layers
    # model.model.decoder.layers
    # layers = find_layers(model,allow_list=(nn.Conv2d,)) 
    
    # module_float_layers = find_layers(float_model,allow_list=(nn.Linear,nn.Conv2d))# float_model.model.decoder.layers
    # float_layers = find_layers(float_model,allow_list=(nn.Conv2d,))
    # checking the layers and float layers:
    # for m1,m2 in zip(module_layers,module_float_layers):
    #     if len(m1.fwd_kwargs)>0:
    #         assert m2.stride == m1.fwd_kwargs['stride'], "The stride of layers in model and float_model are not equal."
    #         assert m2.padding == m1.fwd_kwargs['padding'], "The padding of layers in model and float_model are not equal."
    #         assert m2.dilation == m1.fwd_kwargs['dilation'], "The dilation of layers in model and float_model are not equal."
    #         assert m2.groups == m1.fwd_kwargs['groups'], "The groups of layers in model and float_model are not equal."

    # blocks = find_layers(model,allow_list=(BaseQuantBlock,)) 
    
    dtype = next(iter(model.parameters())).dtype
    
    # torch.cuda.empty_cache()
    
    # 这里需要注意如果卷积后面带个relu或者其他模块的融合，很可能会导致错误。
    # 还有一个很重要的问题，这个BRECQ的是现实并不是计算前对浮点激活值进行量化，反而是输入是量化后的值，输出的值是浮点值，并且被量化为整型，
    # 因此，其与浮点的对齐很奇怪，其输入应该是上一层的输出而不是浮点值，中间一直是整形
    def bias_compensation_imp(float_layers,layers):
        ul2ptq_quantizers = []
        if verbose:
            logger.info("    Preparing to Applying Bias Compensation.")
        # caching the inputs and outputs of each float layer
        float_inputs_outputs = [GetInputsOutputs(layer) for layer in float_layers]
        for m in float_inputs_outputs:
            m.register_hook()
        loss_fn = nn.MSELoss().to(dev)
        loss1s = [0.]*(len(float_inputs_outputs))
        # loss2s = [0.]*(len(float_inputs_outputs))
        if verbose:
            logger.info("    Finding the Best Bias Compensation")  
        for batch_index, (imgs,targets) in enumerate(dataloader):
            # if batch_index*imgs.size(0)>args.num_samples:
            #     break
            imgs = imgs.to(dev)
            _ = float_model(imgs)
            # Got the inputs and outputs of each layer
            for i,(cached_inputs_outputs) in enumerate(float_inputs_outputs):
                inputs = cached_inputs_outputs.inputs
                outputs = cached_inputs_outputs.outputs.detach()
                # get quantized outputs for the same inputs
                q_layer = layers[i]
                q_outputs = q_layer(*inputs)
                # calculate the loss
                loss1s[i] += loss_fn(q_outputs,outputs)
                # ul2ptq quantizer
                if len(ul2ptq_quantizers)<=i:
                    if outputs.dim()==4:
                        channel_size=np.prod(outputs.shape[1:])
                        channel_axis=(1,2,3)
                    if outputs.dim()==3:
                        channel_size=np.prod(outputs.shape[1:])
                        channel_axis=(1,2)
                    elif outputs.dim()==2:
                        channel_size=outputs.shape[1]
                        channel_axis=(1,)
                    ul2ptq_quantizers.append(UL2PTQ3(channel_size=channel_size,
                                                    channel_axis=channel_axis,
                                                    ).to(dev))
                ul2ptq_quantizers[i].update(outputs, q_outputs, post_bias=False)
                # loss2s[i] += loss_fn(q_outputs+\
                #                 ul2ptq_quantizers[i].bias.view(ul2ptq_quantizers[i].bias_shape),
                #                 outputs)
            
        if verbose:
            logger.info("    Applying Bias Compensation.")
        for i,(layer) in enumerate(layers):
            quantizer = ul2ptq_quantizers[i]
            quantize_module_act(layer,quantizer,act_id=0,pre=False,device=dev)
            if verbose:
                logger.info(f"    \t{'Layer-%d' % (i)} "
                            f"loss1={loss1s[i]/len(dataloader):>.5f}")
        #########################TEST####################   
        # 测试一下是不是真的BC生效了
        if DEBUG_BC and verbose:
            losses1 = [0.]*(len(layers)+1)
            losses2 = [0.]*(len(layers)+1)
            for batch_index, (imgs,targets) in enumerate(dataloader):
                # if batch_index*imgs.size(0)>args.num_samples:
                #     break
                print()
                imgs = imgs.to(dev)
                _ = float_model(imgs)
                # Got the inputs and outputs of each layer
                for i,(cached_inputs_outputs) in enumerate(float_inputs_outputs):
                    inputs = cached_inputs_outputs.inputs
                    outputs = cached_inputs_outputs.outputs.detach()
                    # get quantized outputs for the same inputs
                    q_layer = layers[i]
                    q_outputs2 = q_layer(*inputs)
                    qm = q_layer.quantized_act0_post_module
                    q_layer.quantized_act0_post_module = nn.Identity()
                    q_outputs1 = q_layer(*inputs)
                    q_layer.quantized_act0_post_module = qm
                    losses2[i]+=loss_fn(outputs,q_outputs2)
                    losses1[i]+=loss_fn(outputs,q_outputs1)

            logger.info("    Test Bias Compensation Validation")
            for i, _ in enumerate(layers):
                logger.info(f"    \t{'Layer-%d' % (i)} "
                    f"loss1={losses1[i]/len(dataloader):>.5f}, "
                    f"loss2={losses2[i]/len(dataloader):>.5f}")
        #########################END TEST####################   
        # remove all hooks
        for m in float_inputs_outputs:
            m.remove_hook()

        if verbose:
            logger.info("    Bias Compensation End.")  

        del float_inputs_outputs
        float_inputs_outputs = None
        torch.cuda.empty_cache()

        return ul2ptq_quantizers

    def del_bias_compensation_imp(model):
        for i,m in enumerate(model.modules()):
            if hasattr(m,"quantized_act0_post_module"):
                qm = m.quantized_act0_post_module
                m.quantized_act0_post_module = nn.Identity()
                del qm
        if verbose:
            logger.info('Del BC End.')
        torch.cuda.empty_cache()

    # 首先清除以往的bc以防止其影响结果
    del_bias_compensation_imp(model)

    # 应用新的bc补偿精度
    global_ul2ptq_quantizers=[]
    
    allow_list=[]
    if (args.quantize_pos&1):
        allow_list.append(PTQSLQuantConv2d)
    if (args.quantize_pos&2):
        allow_list.append(PTQSLQuantLinear)
    if (args.quantize_pos&4):
        allow_list.append(PTQSLQuantMatMul)
    if len(allow_list)==0:
        raise ValueError("Must Set a quantize position")
    allow_list = tuple(allow_list)
    print(allow_list)
    # allow_list=(PTQSLQuantConv2d,PTQSLQuantLinear,PTQSLQuantMatMul)
    # allow_list=(PTQSLQuantConv2d,PTQSLQuantLinear)
    # allow_list = (PTQSLQuantLinear,)
    # allow_list=(PTQSLQuantMatMul,)
    # allow_list=(PTQSLQuantConv2d,)
    module_layers = find_layers(model,allow_list=allow_list)
    module_float_layers = find_layers(float_model,allow_list=allow_list)
    assert len(module_layers)==len(module_float_layers), "The number of layers in model and float_model are not equal."
    
    # 分段优化以节约内存
    if not args.n_align_layer:
        logger.info('Applying Bias Compensation to Quantized Layers.')
        seg_len=32
        for i in range(0,len(module_layers),seg_len):
            i_upper=min(i+seg_len,len(module_layers))
            global_ul2ptq_quantizers+=bias_compensation_imp(module_float_layers[i:i_upper],module_layers[i:i_upper])
        # global_ul2ptq_quantizers+=bias_compensation_imp(module_float_layers,module_layers)

    if hasattr(model,"blocks"):
        block_layers = list(model.blocks)
        block_float_layers = list(float_model.blocks)
        assert len(block_layers)==len(block_float_layers), "The number of block layers in model and float_model are not equal."
        
        attn_layers = [block.attn for block in block_layers]
        attn_float_layers = [block.attn for block in block_float_layers]
        assert len(attn_layers)==len(attn_float_layers), "The number of block layers in model and float_model are not equal."
        
        mlp_layers = [block.mlp for block in block_layers]
        mlp_float_layers = [block.mlp for block in block_float_layers]
        assert len(mlp_layers)==len(mlp_float_layers), "The number of block layers in model and float_model are not equal."
        
        if args.align_attn:
            logger.info('Applying Bias Compensation to Quantized Attn.')
            global_ul2ptq_quantizers+=bias_compensation_imp(attn_float_layers,attn_layers)
        
        if args.align_mlp:
            logger.info('Applying Bias Compensation to Quantized Mlp.')
            global_ul2ptq_quantizers+=bias_compensation_imp(mlp_float_layers,mlp_layers)
        
        if args.align_blocks:
            logger.info('Applying Bias Compensation to Quantized Blocks.')
            global_ul2ptq_quantizers+=bias_compensation_imp(block_float_layers,block_layers)


    elif hasattr(model,"layers"):
        logger.info('Applying Bias Compensation to Quantized Blocks.')
        
        layers = list(model.layers)
        float_layers = list(float_model.layers)
        assert len(layers)==len(float_layers), "The number of block layers in model and float_model are not equal."
        
        downsample_layers= [layer.downsample for layer in layers if hasattr(layer,"downsample") and (not isinstance(layer.downsample,nn.Identity)) and (layer.downsample is not None)]
        downsample_float_layers= [layer.downsample for layer in float_layers if hasattr(layer,"downsample") and (not isinstance(layer.downsample,nn.Identity)) and (layer.downsample is not None)]
        assert len(downsample_layers)==len(downsample_float_layers), "The number of block layers in model and float_model are not equal."
        
        block_layers = [list(layer.blocks) for layer in layers]
        block_layers = [item for sublist in block_layers for item in sublist]
        block_float_layers = [list(layer.blocks) for layer in float_layers]
        block_float_layers = [item for sublist in block_float_layers for item in sublist]
        assert len(block_layers)==len(block_float_layers), "The number of block layers in model and float_model are not equal."
        
        attn_layers= [layer.attn for layer in block_layers]
        attn_float_layers= [layer.attn for layer in block_float_layers]
        assert len(attn_layers)==len(attn_float_layers), "The number of block layers in model and float_model are not equal."
        
        mlp_layers= [layer.mlp for layer in block_layers]
        mlp_float_layers= [layer.mlp for layer in block_float_layers]
        assert len(mlp_layers)==len(mlp_float_layers), "The number of block layers in model and float_model are not equal."
        
        if args.align_attn:
            logger.info('Applying Bias Compensation to Quantized Attn.')
            global_ul2ptq_quantizers+=bias_compensation_imp(attn_float_layers,attn_layers)
        
        if args.align_mlp:
            logger.info('Applying Bias Compensation to Quantized Mlp.')
            global_ul2ptq_quantizers+=bias_compensation_imp(mlp_float_layers,mlp_layers)
        
        if args.align_blocks:
            logger.info('Applying Bias Compensation to Quantized Downsample Layers.')
            global_ul2ptq_quantizers+=bias_compensation_imp(downsample_float_layers,downsample_layers)

            logger.info('Applying Bias Compensation to Quantized Blocks.')
            global_ul2ptq_quantizers+=bias_compensation_imp(block_float_layers,block_layers)

            logger.info('Applying Bias Compensation to Quantized Layers.')
            global_ul2ptq_quantizers+=bias_compensation_imp(float_layers,layers)

        # global_ul2ptq_quantizers+=bias_compensation_imp(block_float_layers,block_layers)
    
        # block_layers = list(model.layers)
        # block_float_layers = list(float_model.layers)
        # assert len(block_layers)==len(block_float_layers), "The number of block layers in model and float_model are not equal."
        
        # global_ul2ptq_quantizers+=bias_compensation_imp(block_float_layers,block_layers)
    
    if args.n_align_model:
        global_ul2ptq_quantizers+=bias_compensation_imp([float_model],[model])

    return global_ul2ptq_quantizers



def test_all(name, cfg_modifier=lambda x: x, calib_size=32, config_name="PTQ4ViT"):
    quant_cfg = init_config(config_name)
    quant_cfg = cfg_modifier(quant_cfg)

    net = get_net(name)

    g=datasets.ViTImageNetLoaderGenerator(args.datapath,'imagenet',args.batch_size,args.batch_size,16, kwargs={"model":net})
    test_loader=g.test_loader()
    calib_loader=g.calib_loader(num=calib_size,seed=args.seed)

    # print("Before warpper:",net,quant_cfg)
    # acc = test_classification(net,test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"])
    # print("Before quantization acc:",acc)

    wrapped_modules=net_wrap.wrap_modules_in_net(net,quant_cfg)
    # print("After warpper:",net)
    # acc = test_classification(net,test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"])
    # print("After quantization acc:",acc)
    if args.use_bc:
        float_net = copy.deepcopy(net).cpu()
        # apply_bias_compensation(float_net,net,calib_loader,next(net.parameters()).device)
    
    DEBUG = False
    # add timing
    if not args.load_quantized_model and not DEBUG:
        logger.info(f"Calibrate quantized model from {args.nsamples} samples with seed {args.seed}")
        calib_start_time = time.time()
        quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=4) # 16 is too big for ViT-L-16
        quant_calibrator.batching_quant_calib()
        calib_end_time = time.time()
    elif args.load_quantized_model:
        save_name = f"{args.model}-W{args.wbits}A{args.abits}-{args.nsamples}samples-{args.seed}seed-{args.quantizer}.pth"
        save_path = f"{args.save_path}/{save_name}"
        logger.info(f"Loading quantized model from {save_path}")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Quantized model {save_path} not found.")
        # net.load_state_dict(torch.load(save_path))
        # 加载模型
        # net = torch.load(save_path)
        state = torch.load(save_path)
        net.load_state_dict(state['model'])
        for name, module in net.named_modules():
            attrs = state['attrs'].get(name, {})
            for k, v in attrs.items():
                setattr(module, k, v)
    # 尝试保存量化后的模型
    if (not args.load_quantized_model) and args.save_quantized_model:
        save_name = f"{args.model}-W{args.wbits}A{args.abits}-{args.nsamples}samples-{args.seed}seed-{args.quantizer}.pth"
        save_path = f"{args.save_path}/{save_name}"
        # if not os.path.exists(args.save_path):
            # raise FileNotFoundError(f"Quantized model {save_path} not found.")
        os.makedirs(args.save_path,exist_ok=True)
        # 保存模型
        # torch.save(net.state_dict(), save_path)
        state = {
            'model': net.state_dict(),
            'attrs': {name: {k: v for k, v in module.__dict__.items() \
                             if not (k.startswith('_') or isinstance(v, (torch.nn.Module,dict,torch.tensor)))} 
                    for name, module in net.named_modules()}
        }
        print(state)
        # torch.save(net, save_path)
        logger.info(f"Save quantized model to {save_path}")

    if args.test_quantized_model and not DEBUG:
        quantized_acc = test_classification(net,test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"])
    else:
        quantized_acc = -1

    
    if args.use_bc:
        # net = net.cpu()
        # # 清空显存
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        float_net = float_net.cuda()
        bc_calib_loader=g.bc_calib_loader(num=args.n_bc_samples,seed=args.bc_seed) # 设置batchsize=1防止显存溢出
        # 如果启用启动搜索，那么将自动搜索2^5=32种方案中的最优方案并列出来，
        # 即args.n_align_layer, args.align_attn, args.align_mlp,args.align_blocks, args.n_align_model这5个值的组合
        if args.gsfb: 
            bak_config = [args.n_align_layer, args.align_attn, args.align_mlp,args.align_blocks, args.n_align_model]
            gsfb_accs = {(0,0,0,0,0):quantized_acc}
            def extract_5bit(x):
                y = []
                for i in range(5):
                    y.append(x&1)
                    x = x>>1
                return y
            for i in range(1, 2**5):
                # parse int to 5 binary value according to te bit 
                key = tuple(extract_5bit(i))
                align_layer, args.align_attn, args.align_mlp,args.align_blocks, align_model = key
                args.n_align_layer = not align_layer
                args.n_align_model = not align_model
                logger.info(f"BC Strategies: align_layer:{align_layer}, "
                            f"align_attn: {args.align_attn}, "
                            f"align_mlp: {args.align_mlp}, "
                            f"align_blocks: {args.align_blocks}, "
                            f"align_model: {align_model}")
                apply_bias_compensation(float_net,net,bc_calib_loader,next(net.parameters()).device,False)
                bc_acc = test_classification(net,test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"])
                gsfb_accs[key] = bc_acc
            
            logger.info(f"model: {name}")
            logger.info(f"calibration size: {calib_size}")
            logger.info(f"bc calibration size: {args.n_bc_samples}")
            logger.info(f"bc calibration seed: {args.bc_seed}")
            logger.info(f"bit settings: {quant_cfg.bit}")
            logger.info(f"Grid Search Results (align_layer, align_attn, align_mlp,align_blocks, align_model): ")
            for key,value in gsfb_accs.items():
                logger.info(f"{key}: {value}")

            logger.info(f"Grid Search End\n")

        elif args.test_pose:
            raw_num = sum(p.numel() for p in net.parameters())
            # test the impact of bias compensation positions on the accuracy
            pose_set=[1,2,4,3,5,6,7]
            pose_name=["Conv","Linear","MatMul",
                       "Conv+Linear","Conv+MatMul","Linear+MatMul","Conv+Linear+MatMul"]
            params = {key:0.0 for key in pose_name}
            accs = {key:0.0 for key in pose_name}
            for i,(pose,key) in enumerate(zip(pose_set,pose_name)):
                args.quantize_pos = pose
                logger.info(f"BC Strategies: {key}")
                apply_bias_compensation(float_net,net,bc_calib_loader,next(net.parameters()).device,False)
                bc_acc = test_classification(net,test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"])
                logger.info(f"BC Accuracy: {bc_acc}")
                accs[key] = bc_acc
                params[key] = sum(p.numel() for p in net.parameters())
            logger.info(f"model: {name}")
            logger.info(f"calibration size: {calib_size}")
            logger.info(f"bc calibration size: {args.n_bc_samples}")
            logger.info(f"bc calibration seed: {args.bc_seed}")
            logger.info(f"bit settings: {quant_cfg.bit}")
            logger.info(f"Parameter number before BC: {raw_num}")
            for key in pose_name:
                logger.info(f"Employing {key} acc:{accs[key]:.4f} #params: {params[key]}")
            logger.info(f"Test Bias Compensation Position End\n")

        else:
            num = sum(p.numel() for p in net.parameters())
            apply_bias_compensation(float_net,net,bc_calib_loader,next(net.parameters()).device)
            if not DEBUG:
                bc_acc = test_classification(net,test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"])
            else:
                bc_acc = -1
            a_num = sum(p.numel() for p in net.parameters())
            logger.info(f"model: {name}")
            logger.info(f"calibration size: {calib_size}")
            logger.info(f"bc calibration size: {args.n_bc_samples}")
            logger.info(f"bc calibration seed: {args.bc_seed}")
            logger.info(f"bit settings: {quant_cfg.bit}")
            logger.info(f"quantized accuracy: {quantized_acc}")
            logger.info(f"bc accuracy: {bc_acc}\n")
            logger.info(f"calibration time: {(calib_end_time-calib_start_time)/60}min")
            logger.info(f"Parameter number before BC: {num}")
            logger.info(f"Parameter number after BC: {a_num}\n")
    else:
        logger.info(f"model: {name}")
        logger.info(f"calibration size: {calib_size}")
        logger.info(f"bit settings: {quant_cfg.bit}")
        logger.info(f"config: {config_name}")
        logger.info(f"ptqsl_conv2d_kwargs: {quant_cfg.ptqsl_conv2d_kwargs}")
        logger.info(f"ptqsl_linear_kwargs: {quant_cfg.ptqsl_linear_kwargs}")
        logger.info(f"ptqsl_matmul_kwargs: {quant_cfg.ptqsl_matmul_kwargs}")
        logger.info(f"calibration time: {(calib_end_time-calib_start_time)/60}min")
        logger.info(f"quantized accuracy: {quantized_acc}")

    # Applying BC to imptove accuracy


class cfg_modifier():
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self,name,value)

    def __call__(self, cfg):
        # bit setting
        cfg.bit = self.bit_setting
        cfg.w_bit = {name: self.bit_setting[0] for name in cfg.conv_fc_name_list}
        cfg.a_bit = {name: self.bit_setting[1] for name in cfg.conv_fc_name_list}
        cfg.A_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}
        cfg.B_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}

        # conv2d configs
        cfg.ptqsl_conv2d_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_conv2d_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_conv2d_kwargs["metric"] = self.metric
        cfg.ptqsl_conv2d_kwargs["init_layerwise"] = False

        # linear configs
        cfg.ptqsl_linear_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_linear_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_linear_kwargs["n_a"] = self.linear_ptq_setting[2]
        cfg.ptqsl_linear_kwargs["metric"] = self.metric
        cfg.ptqsl_linear_kwargs["init_layerwise"] = False

        # matmul configs
        cfg.ptqsl_matmul_kwargs["metric"] = self.metric
        cfg.ptqsl_matmul_kwargs["init_layerwise"] = False

        return cfg

if __name__=='__main__':
    args = parse_args()

    # set_seed(args.seed)
    seed_all(args.seed)
    
    bc_st = ""
    if args.use_bc:
        bc_st += f"{'-use-bc' if args.use_bc else ''}"
        f"{'-debug-bc' if args.debug_bc else ''}"
        if args.gsfb: 
            bc_st += "-gsfb"   
        else: 
            bc_st += f"{'-align-atten' if args.align_attn else ''}"
            f"{'-align-mlp' if args.align_mlp else ''}"
            f"{'-align-blocks' if args.align_blocks else ''}"
            f"{'-n-align-layer' if args.n_align_layer else ''}"
            f"{'-n-align-model' if args.n_align_model else ''}"

    logger.info(f"\n########PTQ4ViT Quantizing "
                f"{args.model}-W{args.wbits}A{args.abits}-{args.nsamples}samples{bc_st}"
                "########\n")
    os.makedirs(args.log_root,exist_ok=True)
    logging_file = os.path.join(args.log_root, "PTQ4ViT-test-"
                    f"{args.model}-W{args.wbits}A{args.abits}-{args.nsamples}samples{bc_st}.log")
    file_handler = logging.FileHandler(logging_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"# logging file save to {logging_file}")
    if not args.verbose:
        logger.info(f"## Stop output to screen, please see the log in {logging_file}")
        logger.removeHandler(console_handler)

    cadicate_names = [
        "vit_tiny_patch16_224",
        # "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        "vit_small_patch32_224",
        "vit_small_patch16_224", # VIT-S
        "vit_base_patch16_224", # VIT-B
        "vit_base_patch16_384", # VIT-B*

        "deit_tiny_patch16_224", # Deit-T
        "deit_small_patch16_224", # Deit-S
        "deit_base_patch16_224", # Deit-B
        "deit_base_patch16_384", # Deit-B*

        "swin_tiny_patch4_window7_224", # Swin-T
        "swin_small_patch4_window7_224", # Swin-S
        "swin_base_patch4_window7_224", # Swin-B
        "swin_base_patch4_window12_384", # Swin-B*
        ]
    if args.model not in cadicate_names:
        raise ValueError(f"model name {args.model} is not in {cadicate_names}")
    
    names = [args.model,]
    metrics = ["hessian"]
    linear_ptq_settings = [(1,1,1)] # n_V, n_H, n_a
    # calib_sizes = [32,128]
    calib_sizes = [args.nsamples,]
    # bit_settings = [(8,8), (6,6),(4,4)] # weight, activation
    bit_settings = [(args.wbits,args.abits),]
    # config_names = ["PTQ4ViT", "BasePTQ"]
    config_names = [args.quantizer,]

    cfg_list = []
    for name, metric, linear_ptq_setting, calib_size, bit_setting, config_name in product(names, metrics, linear_ptq_settings, calib_sizes, bit_settings, config_names):
        cfg_list.append({
            "name": name,
            "cfg_modifier":cfg_modifier(linear_ptq_setting=linear_ptq_setting, metric=metric, bit_setting=bit_setting),
            "calib_size":calib_size,
            "config_name": config_name
        })
    
    if args.multiprocess:
        multiprocess(test_all, cfg_list, n_gpu=args.n_gpu)
    else:
        for cfg in cfg_list:
            logger.info(f"##### Quantizing {cfg['name']} with {cfg['calib_size']} samples with {cfg['config_name']}({cfg['cfg_modifier'].bit_setting}) ####")
            test_all(**cfg)