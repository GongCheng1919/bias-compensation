import os
os.environ['http_proxy'] = 'http://10.134.55.178:4780'
os.environ['https_proxy'] = 'http://10.134.55.178:4780'
data_cache_dir = f"./cache_dir"

GPTQ_PATH = f"./submodules/gptq"
import sys
sys.path.append(GPTQ_PATH)
sys.path.append("./")

import math
import time

import torch
import torch.nn as nn
import transformers
import copy

from gptq import * 
from modelutils import *
from quant import *

from bias_compensation.quantize.quantize import quantize_module_weight, quantize_module_act,quantize_model
from bias_compensation.quantizers.ul2ptq import UL2PTQ, UL2PTQ2, UL2PTQ3
from bias_compensation.utils.general_utils import (
    freeze_all_modules,
    enable_training_modules,
    stat_trainable_params,
    GetInputsOutputs
)
import copy

def get_bloom(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import BloomForCausalLM
    model_folder = f"{data_cache_dir}/models/{model}"
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        model = BloomForCausalLM.from_pretrained(model, torch_dtype='auto')
        model.save_pretrained(model_folder)
    else:
        model = BloomForCausalLM.from_pretrained(model_folder, torch_dtype='auto')

    model.seqlen = 2048
    return model

@torch.no_grad()
def apply_bias_compensation(float_model, model, dataloader, dev):
    DEBUG_BC = args.debug_bc

    print('Starting BC ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h
    float_layers = float_model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    print('Ready.')

    ul2ptq_quantizers = {}

    for i in range(len(layers)):
        if DEBUG_BC:
            print("\nPreparing to Applying Bias Compensation for Layer-%d."%i)  
        layer = layers[i].to(dev)
        float_layer = float_layers[i].to(dev)

        subset = find_layers(layer)
        float_subset = find_layers(float_layer)
        ul2ptq = {}

        if args.small_bias_vector:
            ul2ptq[f"layer-{i}"]=UL2PTQ3(channel_size=1024,
                                channel_axis=(-1,),
                                ).to(dev)
        else:
            ul2ptq[f"layer-{i}"]=UL2PTQ3(channel_size=2048*1024,
                                channel_axis=(-2,-1),
                                ).to(dev)
        for name in subset:
            if args.small_bias_vector:
                ul2ptq[name] = UL2PTQ3(channel_size=subset[name].out_features,
                                    channel_axis=(-1,),
                                    ).to(dev)
            else:
                ul2ptq[name] = UL2PTQ3(channel_size=2048*subset[name].out_features,
                                    channel_axis=(-2,-1),
                                    ).to(dev)

        float_inputs_outputs=[GetInputsOutputs(float_subset[name]) for name in float_subset]
        
        for m in float_inputs_outputs:
            m.register_hook()
        
        if DEBUG_BC:
            print("Finding the Best Bias Compensation for Layer-%d."%i)  
        loss_fn = nn.MSELoss().to(dev)
        loss1s = [0.]*(len(ul2ptq)+1)
        # loss2s = [0.]*(len(ul2ptq)+1)
        for j in range(args.nsamples):
            outs[j] = float_layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            qouts = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            loss1s[-1] += loss_fn(qouts,outs[j].unsqueeze(0))
            for k, (name,q_layer) in enumerate(subset.items()):
                f_inputs = float_inputs_outputs[k].inputs[0].detach()
                f_outputs = float_inputs_outputs[k].outputs[0].detach().unsqueeze(0)
                q_outputs = q_layer(f_inputs)
                ul2ptq[name].update(f_outputs, q_outputs, post_bias=False)
                loss1s[k] += loss_fn(q_outputs,f_outputs) # 计算总损失
                # loss2s[k] += loss_fn(q_outputs+\
                #                     ul2ptq[name].bias.view(ul2ptq[name].bias_shape),
                #                     f_outputs) # 计算总损失
        
        if DEBUG_BC:
            print("Applying Bias Compensation to Layer-%d."%i)  
            print(f"\t{'transformer.h.%d.%s' % (i, 'all')} "
                    f"loss w/o bc={loss1s[-1]/args.nsamples:>.5f}, ")
        for k, (name,q_layer) in enumerate(subset.items()):
            if DEBUG_BC:
                print(f"\t{'transformer.h.%d.%s' % (i, name)} "
                    f"loss w/o bc={loss1s[k]/args.nsamples:>.5f} ")
                    # f"loss2={loss2s[k]/args.nsamples:>.5f}")
            quantize_module_act(q_layer,ul2ptq[name],act_id=0,pre=False)
            # q_layer.quantized_act0_post_module = ul2ptq[name]
            ul2ptq_quantizers['transformer.h.%d.%s' % (i, name)] = ul2ptq[name]
        #########################TEST####################   
        # 测试一下是不是真的BC生效了
        losses1 = [0.]*(len(ul2ptq)+1)
        losses2 = [0.]*(len(ul2ptq)+1)
        for j in range(args.nsamples):
            if DEBUG_BC:
                outs[j] = float_layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            qouts = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            losses1[-1] += loss_fn(qouts,outs[j].unsqueeze(0))
            ul2ptq[f"layer-{i}"].update(outs[j].unsqueeze(0), 
                                        qouts, 
                                        post_bias=False)
            if DEBUG_BC:
                for k, (name,q_layer) in enumerate(subset.items()):
                    f_inputs = float_inputs_outputs[k].inputs[0].detach()
                    f_outputs = float_inputs_outputs[k].outputs[0].detach().unsqueeze(0)
                    q_outputs2 = q_layer(f_inputs)
                    qm = q_layer.quantized_act0_post_module
                    q_layer.quantized_act0_post_module = nn.Identity()
                    q_outputs1 = q_layer(f_inputs)
                    q_layer.quantized_act0_post_module = qm
                    losses2[k]+=loss_fn(f_outputs,q_outputs2)
                    losses1[k]+=loss_fn(f_outputs,q_outputs1)

        quantize_module_act(layer,ul2ptq[f"layer-{i}"],act_id=0,pre=False)
        ul2ptq_quantizers['transformer.h.%d.%s' % (i, 'all')] = ul2ptq[f"layer-{i}"]
        if DEBUG_BC:
            print("Layer-%d's Test Bias Compensation Validation"%i)
            print(f"\t{'transformer.h.%d.%s' % (i, 'all')} "
                    f"loss w bc={losses1[-1]/args.nsamples:>.5f}, ")
            for k, (name,q_layer) in enumerate(subset.items()):
                print(f"\t{'transformer.h.%d.%s' % (i, name)} "
                    f"loss w/o bc={losses1[k]/args.nsamples:>.5f}, "
                    f"loss w bc={losses2[k]/args.nsamples:>.5f}")

        #########################END TEST####################   
        # remove all hooks
        for m in float_inputs_outputs:
            m.remove_hook()

        if DEBUG_BC:
            print("Layer-%d's Bias Compensation End."%i)  

        layers[i] = layer.cpu()
        float_layers[i] = float_layer.cpu()
        del float_inputs_outputs
        float_inputs_outputs = None
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return ul2ptq_quantizers

@torch.no_grad()
def bloom_sequential(model, dataloader, dev, means=None, stds=None):
    # Fake quantization
    # import sys
    # sys.path.append(f"{os.environ['HOME']}/notebooks/llm-quant")
    # def tmp_quantizer(input):
    #     max_val = torch.max(torch.abs(input))
    #     bitwidth = args.wbits
    #     qnum = 2**(bitwidth-1)-0.5
    #     alpha =  qnum/max_val
    #     fixed = input*alpha-0.5
    #     fixed = fixed.to(torch.int).to(input.dtype)+0.5
    #     return fixed/alpha
    # for m,n in model.named_modules():
    #     if isinstance(n,nn.Linear):
    #         n.weight.data = tmp_quantizer(n.weight.data)
    # return

    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
        for h in handles:
            h.remove()

        for name in subset:
            if args.debug_bc:
                print(i, name)
                print('Quantizing ...')
            gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize)
            quantizers['transformer.h.%d.%s' % (i, name)] = gptq[name].quantizer
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers

@torch.no_grad()
def bloom_eval(model, testenc, dev):
    print('Evaluation...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    elaspsed_time = 0
    for i in range(len(layers)):
        # print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                # 在这里添加groupsize的支持试试
                if args.groupsize != -1:
                    import copy
                    # W = W.clone()
                    # reshape to 2 dim
                    if isinstance(subset[name], nn.Conv2d):
                        W = W.flatten(1)
                    if isinstance(subset[name], transformers.Conv1D):
                        W = W.t()
                    # W = W.float()
                    # grouping
                    for j in range(0, W.shape[1], args.groupsize):
                        group_quantizer = copy.deepcopy(quantizer)
                        top_index = min(j + args.groupsize,W.shape[1])
                        group_quantizer.find_params(W[:, j:top_index], weight=True)
                        # quantizing
                        W[:, j:top_index] = quantize(
                            W[:, j:top_index], group_quantizer.scale, group_quantizer.zero, group_quantizer.maxq
                        )
                    # reset weight
                    if isinstance(subset[name], transformers.Conv1D):
                        W = W.t()
                    subset[name].weight.data = W.reshape(subset[name].weight.shape).to(subset[name].weight.data.dtype)
                    # subset[name].weight.data.copy_(W)

                else:
                    quantizer.find_params(W, weight=True)
                    subset[name].weight.data = quantize(
                        W, quantizer.scale, quantizer.zero, quantizer.maxq
                    ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            start_time=time.perf_counter()
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            torch.cuda.synchronize() 
            elaspsed_time+=time.perf_counter()-start_time

        layers[i] = layer.cpu() 
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.transformer.ln_f = model.transformer.ln_f.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)

        start_time=time.perf_counter()
        hidden_states = model.transformer.ln_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        torch.cuda.synchronize()
        elaspsed_time+=time.perf_counter()-start_time

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print("Perplexity = %.2f"%ppl.item())
    print("Token_num = ", nsamples * model.seqlen)
    print("Evaluate_time = %.4fs"%elaspsed_time)
    print("Tokens/s = %.2fs"%(nsamples * model.seqlen/elaspsed_time))

    model.config.use_cache = use_cache


def bloom_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='BLOOM model to load; pass `bigscience/bloom-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval'
    )
    parser.add_argument(
        '--use-bc', action='store_true',
        help='Whether to apply the bias compensation in model quantization'
    )
    parser.add_argument(
        '--debug-bc', action='store_true',
        help='Whether to show the bias compensation error reduce'
    )
    parser.add_argument(
        '--small-bias-vector', action='store_true',
        help='Whether to show the bias compensation error reduce'
    )

    args = parser.parse_args()

    configs=f"{args.model}-{args.dataset}-wbits{args.wbits}-groupsize{args.groupsize}-small_bias_vector{args.small_bias_vector}-use_bc{args.use_bc}"
    print("GPTQ+BC quantizing configs:",configs)

    model = get_bloom(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen,load_train=True
    )
    
    num = sum(p.numel() for p in model.parameters())
    print(f"Parameter number of {args.model}: {num}")

    if args.wbits < 16 and not args.nearest:
        
        if args.use_bc:
            float_model = copy.deepcopy(model)
        
        # 量化模型
        tick = time.time()
        quantizers = bloom_sequential(model, dataloader, DEV)
        print("GPTQ quantize time:",time.time() - tick)
        # 对模型应用UL2PTQ量化
        if args.use_bc:
            num = sum(p.numel() for p in model.parameters())
            tick = time.time()
            apply_bias_compensation(float_model, model, dataloader, DEV)
            print("BC time:",time.time() - tick)
            del float_model
            float_model=None
            torch.cuda.empty_cache()
            a_num = sum(p.numel() for p in model.parameters())
            print(f"Parameter number of GPTQ: {num}")
            print(f"Parameter number of GPTQ+BC: {a_num}\n")

    datasets = ['wikitext2', 'ptb', 'c4'] 
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets: 
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen,load_train=False
        )
        print(dataset)
        bloom_eval(model, testloader, DEV)

    if args.save:
        bloom_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)
