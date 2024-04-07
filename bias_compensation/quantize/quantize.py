import torch
from torch import nn
import copy

def set_function_name(name):
    def decorator(func):
        func.__name__ = name
        return func
    return decorator
    
def quantize_module_weight(module,
                           quantize_method,
                           weight_name,
                           PTQ=False,
                           remove_all_weight_hooks=True,
                           device=None):
    ''' quantize the module parameters with the attr, such as weight, bias, 
    This function tries to quantize the module weight by replacing the module weight with a quantized one. 
    If necessary, this function will store a float weight for QAT, but it will double the model size.
    @ module: nn.Module type, module whose weight need to be quantized
    @ quantize_method: nn.Module type, quantizer for the weight
    @ weight_name: string type, the weight name in module
    @ PTQ: bool type, 
        if PTQ=True, we just change the original weight
        if PTQ=False, we should store the original float weight and map it to a quantized one
    @ return: bool type
    # 计算前向传播
    # 注意,在这里的部分,实际上只会调用weight进行前向传播,不需要再次对float_weight量化到weight,
    # 或者说在每次前向传播的时候,我们都不再需要将float_weight重新量化到weight,
    # 这就导致无法在float_weight上累积更新.
    # 即使通过保存中间过程梯度来累积float weight的更新,
    # 更新后的float_weight也无法前向更新到weight变量上
    # 因此我们还必须把计算的这一步通过hook函数嵌入到forward之前,也就是提前做权值的量化才行
    '''
    assert(isinstance(module,nn.Module))
    # assert(isinstance(quantize_method,nn.Module))
    if isinstance(quantize_method,nn.Module):
        if device is None:
            if next(module.parameters()) is not None:
                device = next(module.parameters()).device
            else:
                device = torch.device("cuda")

        quantize_method = quantize_method.to(device)

    # quantize configs
    original_name = weight_name
    float_name = "float_"+weight_name
    quantize_module_name = "quantized_"+weight_name+"_module"
    quantize_hook_name = "quantize_"+weight_name+"_hook"

    # verify if there has registered a weight quantizer,
    # if so, just update the quantize_module
    has_registered_weight_quantizer = False
    if hasattr(module,float_name) and \
        hasattr(module,quantize_module_name) and \
        len(module._forward_pre_hooks)>0:
        for key in list(module._forward_pre_hooks.keys()):
            val = module._forward_pre_hooks[key]
            if val.__name__ == quantize_hook_name:
                has_registered_weight_quantizer=True
                break
    if has_registered_weight_quantizer:
        setattr(module,quantize_module_name,quantize_method)
        # if the weight is parameter, delete it and replace it with a quantized one
        if hasattr(module,original_name) and (not PTQ):
            delattr(module, original_name)
        return True

    @set_function_name(quantize_hook_name)
    def hook_pre(module, input):
        quantize_method = getattr(module,quantize_module_name)
        float_weight = getattr(module,float_name)
        quantized_weight = quantize_method(float_weight)
        setattr(module,original_name,quantized_weight)
        return input
        
    if hasattr(module,weight_name):
        module_weight = getattr(module,weight_name)
        assert not callable(module_weight) and isinstance(module_weight,torch.Tensor)
        
        setattr(module,quantize_module_name,quantize_method)
        # PTQ量化
        if PTQ:
            quantized_weight = quantize_method(module_weight)
            if isinstance(module_weight,nn.Parameter):
                quantized_weight = nn.Parameter(quantized_weight)
            delattr(module, weight_name)
            setattr(module,weight_name,quantized_weight)
            return True

        # QAT量化
        elif not hasattr(module,float_name):
            clone_weight = module_weight.clone()
            if isinstance(module_weight,nn.Parameter):
                clone_weight = nn.Parameter(clone_weight)

            # quantized_weight = quantize_method(clone_weight)
            delattr(module, weight_name)
            setattr(module,float_name,clone_weight)
        else:
            pass
        
        # 嵌入hook,在forward之前将浮点权值量化为整型权值
        register = module.register_forward_pre_hook
        hooks = module._forward_pre_hooks
        hook_func = hook_pre
        if remove_all_weight_hooks:
            if len(module._forward_pre_hooks)>0:
                for key in list(module._forward_pre_hooks.keys()):
                    val = module._forward_pre_hooks[key]
                    if val.__name__ == quantize_hook_name:
                        # key.remove()
                        # print("delete hook",val,val.__name__)
                        del module._forward_pre_hooks[key]

        # Register the weight quantization hook
        register(hook_func)
        
        return True
            
    else:
        raise ValueError(f"There is no parameter named {weight_name} in {module.__class__.__name__}")
        return False

def quantize_module_act(module,
                        quantize_method,
                        act_id=0,
                        pre=True,
                        remove_all_act_hooks=True,
                        device=None):
    '''
    This function tries to quantize the module's input/output by hook.
    It embeds the quantize_method before or after the forward function to implement activation quantization.
    @ module: nn.Module type, module whose weight need to be quantized
    @ quantize_method: nn.Module type, quantizer for the weight
    @ act_id: int type, default=0, to process module with multiple inputs/outputs
    @ pre: bool type, 
        if pre=True, we add the quantization before forward and quantize the input
        if pre=False, we add a hook after the forward and quantize the output
    @ return: bool type
    '''
    assert(isinstance(module,nn.Module))
    # assert(isinstance(quantize_method,nn.Module))
    if isinstance(quantize_method,nn.Module):
        if device is None:
            if next(module.parameters()) is not None:
                device = next(module.parameters()).device
            else:
                device = torch.device("cuda")
        quantize_method = quantize_method.to(device)

    # quantize configs
    quantize_hook_name=f"quantize_act{act_id}_{'pre' if pre else 'post'}_hook"
    quantize_module_name = f"quantized_act{act_id}_{'pre' if pre else 'post'}_module"

    def to_list(data):
        if isinstance(data, list):
            return data
        elif isinstance(data, (tuple, set)):
            return list(data)
        elif isinstance(data, dict):
            return list(data.values())
        else:
            return [data]
    
    @set_function_name(quantize_hook_name)
    def hook_pre(module, input):
        input = to_list(input)[act_id]
        quantize_method = getattr(module,quantize_module_name)
        quantized_input = quantize_method(input)
        return quantized_input

    @set_function_name(quantize_hook_name)
    def hook_post(module,input,output):
        output = to_list(output)[act_id]
        quantize_method = getattr(module,quantize_module_name)
        try:
            quantized_output = quantize_method(output)
        except TypeError:
            quantized_output = quantize_method(input,output)
        return quantized_output
        
    # register_forward_hook, register_forward_pre_hook
    if pre:
        register = module.register_forward_pre_hook
        hooks = module._forward_pre_hooks
        hook_func = hook_pre
    else:
        register = module.register_forward_hook
        hooks = module._forward_hooks
        hook_func = hook_post

    # verify if there has registered a act quantizer,
    has_registered_act_quantizer = False
    if hasattr(module,quantize_module_name) and \
        len(hooks)>0:
        for key in list(hooks.keys()):
            val = hooks[key]
            if val.__name__ == quantize_hook_name:
                has_registered_act_quantizer=True
                break
    # if so, just update the quantize_module
    if has_registered_act_quantizer:
        setattr(module,quantize_module_name,quantize_method)
        return True

    setattr(module,quantize_module_name,quantize_method)
    
    if remove_all_act_hooks:
        if len(hooks)>0:
            for key in list(hooks.keys()):
                val = hooks[key]
                if val.__name__ == quantize_hook_name:
                    del hooks[key]

        # if pre and len(module._forward_pre_hooks)>0:
        #     for key in list(module._forward_pre_hooks.keys()):
        #         val = module._forward_pre_hooks[key]
        #         if val.__name__ == quantize_hook_name:
        #             del module._forward_pre_hooks[key]

        # if (not pre) and len(module._forward_hooks)>0:
        #     for key,val in list(module._forward_hooks.keys()):
        #         val = module._forward_hooks[key]
        #         if val.__name__ == quantize_hook_name:
        #             del module._forward_hooks[key]
    
    # Register _forward_pre_hooks and _forward_hooks    
    register(hook_func)

    return True

'''
Usage:
    weight_quant = FixedPointQunatization(2,False)
    act_quant = TernaryQuantization()
    quantize_module_weight(model.features[0],weight_quant,"weight",PTQ=False)
    quantize_module_act(model.features[0],act_quant,act_id=0,pre=False)
    # output feature for verification
    model.features[0]
    # change the weight quantizer
    model.features[0].quantized_weight_module=TernaryQuantization()
    # plot the quantized weight distribution
    _=plt.hist(model.features[0].weight.data.numpy().reshape(-1),bins=50)
'''

def to_list(data):
    if isinstance(data, list):
        return data
    elif isinstance(data, (tuple, set)):
        return list(data)
    elif isinstance(data, dict):
        return [data]
    else:
        return [data]
    
def quantize_model(model,
                   allow_list,
                   quantize_weight_name,
                   weight_quantizer_cls=None,weight_quantizer_kwargs={},
                   act_pre_quantizer_cls_list=None,act_pre_quantizer_kwargs_list=[],
                   act_post_quantizer_cls_list=None,act_post_quantizer_kwargs_list=[],
                   PTQ=False,
                   in_place=True,
                   device=None):
    '''
    Quantizes the given model by applying weight and activation quantization to the specified modules.

    Args:
        model (nn.Module): The model to be quantized.
        allow_list (type or tuple): The type or tuple of types of modules to be quantized.
        quantize_weight_name (str): The name of the weight attribute to be quantized.
        weight_quantizer_cls (class, optional): The weight quantizer class to be used. Defaults to None.
        weight_quantizer_kwargs (dict, optional): Additional keyword arguments for the weight quantizer class. Defaults to {}.
        act_pre_quantizer_cls_list (list, optional): The list of activation pre-quantizer classes to be used. Defaults to None.
        act_pre_quantizer_kwargs_list (list, optional): The list of additional keyword arguments for the activation pre-quantizer classes. Defaults to [].
        act_post_quantizer_cls_list (list, optional): The list of activation post-quantizer classes to be used. Defaults to None.
        act_post_quantizer_kwargs_list (list, optional): The list of additional keyword arguments for the activation post-quantizer classes. Defaults to [].
        PTQ (bool, optional): Flag indicating whether to use post-training quantization. Defaults to False.
        in_place (bool, optional): Flag indicating whether to perform quantization in-place or return a new quantized model. Defaults to True.

    Returns:
        nn.Module: The quantized model.

    '''
    if weight_quantizer_cls is None \
        and act_pre_quantizer_cls_list is None \
        and act_post_quantizer_cls_list is None:
        return model
    
    if not in_place:
        model = copy.deepcopy(model)
    if act_pre_quantizer_cls_list is not None:
        act_pre_quantizer_cls_list = to_list(act_pre_quantizer_cls_list)
        act_pre_quantizer_kwargs_list = to_list(act_pre_quantizer_kwargs_list)
    if act_post_quantizer_cls_list is not None:
        act_post_quantizer_cls_list = to_list(act_post_quantizer_cls_list)
        act_post_quantizer_kwargs_list = to_list(act_post_quantizer_kwargs_list)
        
    # step 1: get the module list for quantization
    module_list = []
    for n,m in model.named_modules():
        if isinstance(m,allow_list):
            # print(n)
            module_list.append(m)
    # step 2: select the quantizers for the module list
    if weight_quantizer_cls is not None:
        weight_quantizers = [weight_quantizer_cls(**weight_quantizer_kwargs) for _ in range(len(module_list))]
    
    if act_pre_quantizer_cls_list is not None:
        act_pre_quantizers_list = [[act_pre_quantizer_cls(**act_pre_kwargs) \
                            for act_pre_quantizer_cls, act_pre_kwargs in zip(act_pre_quantizer_cls_list,act_pre_quantizer_kwargs_list)] \
                                for _ in range(len(module_list))]

    if act_post_quantizer_cls_list is not None:
        # act_post_quantizers = [act_post_quantizer_cls(**act_post_quantizer_kwargs) for _ in range(len(module_list))]
        act_post_quantizers_list = [[act_post_quantizer_cls(**act_post_kwargs) \
                            for act_post_quantizer_cls, act_post_kwargs in zip(act_post_quantizer_cls_list,act_post_quantizer_kwargs_list)] \
                                for _ in range(len(module_list))]
    
    # step 3: quantize modules
    for i in range(len(module_list)):
        # quantize weight
        if weight_quantizer_cls is not None:
            quantize_module_weight(module_list[i],weight_quantizers[i],quantize_weight_name,PTQ=PTQ,device=device)
        # quantize input
        if act_pre_quantizer_cls_list is not None:
            for j in range(len(act_pre_quantizers_list[i])):
                quantize_module_act(module_list[i],act_pre_quantizers_list[i][j],act_id=j,pre=True,device=device)
        # quantize output (can use input)
        if act_post_quantizer_cls_list is not None:
            for j in range(len(act_post_quantizers_list[i])):
                quantize_module_act(module_list[i],act_post_quantizers_list[i][j],act_id=j,pre=False,device=device)
            
    return model

'''
Usage:
    model=VGG7()

    # add quantizing modules
    allow_list = (nn.Conv2d)
    quantize_weight_name = "weight"
    allow_list=(torch.nn.Conv2d,)
    weight_quantizer_cls=nn.Identity
    weight_quantizer_kwargs={}
    act_pre_quantizer_cls=nn.Identity
    act_pre_quantizer_kwargs={}
    act_post_quantizer_cls=nn.Identity
    act_post_quantizer_kwargs={}

    q_model = quantize_model(q_model,allow_list,quantize_weight_name,
                            weight_quantizer_cls,weight_quantizer_kwargs,
                            act_pre_quantizer_cls,act_pre_quantizer_kwargs,
                            act_post_quantizer_cls,act_post_quantizer_kwargs,
                            )
    print(q_model)

    # update quantizing modules
    for n,m in q_model.named_modules():
    if isinstance(m,allow_list):
        u = UL2PTQ(m.out_channels,(1,1)) # a new quantizer
        m.quantized_act0_post_module = u.to(device)
    print(q_model)
'''

