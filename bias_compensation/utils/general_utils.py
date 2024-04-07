

def freeze_all_modules(model):
    for module in model.modules():
        module.eval()
    for param in model.parameters():
        param.requires_grad = False

def enable_training_modules(model,enable_list):
    for n,m in model.named_modules():
        if isinstance(m,enable_list):
            for param in m.parameters():
                param.requires_grad = True

def stat_trainable_params(model):
    trainable_param_num = 0
    total_param_num = 0
    for n,p in model.named_parameters():
        print(n,end=" ")
        total_param_num+=p.numel()
        if p.requires_grad:
            print("Trainable ", p.numel())
            trainable_param_num += p.numel()
    #             # print(n,param.requires_grad)
        else:
            print("Frozen ", p.numel())
    print(f"trainable_param_num is {trainable_param_num}/{total_param_num}")

# 将对应层的inputs分别输入对应模块获取outputs，然后计算两个outputs的误差用于调整参数。
class GetInputsOutputs:
    def __init__(self, module):
        self.module = module
        self.inputs = None
        self.outputs = None
        self.hook=None

    def register_hook(self):
        def hook_fn(m, inputs, outputs):
            # print(m)  # 打印模型层的信息
            # print(inputs)  # 打印输入
            # print(outputs)  # 打印输出
            self.inputs = inputs
            self.outputs = outputs
        if self.hook is not None:
            self.hook.remove()
        self.hook = self.module.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook=None