import torch

class TernaryQuantizationFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        # print("Apply ternary quantization in Forward")
        output = scale*torch.clamp(torch.round(input/scale),-1,1)
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        # print("Apply ternary quantization in Backward")
        grad_input = None
        grad_input = grad_output
        return grad_input, None

class TernaryQuantization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bitwidth = 2
        self.scale = 1
        self.quantize_func = TernaryQuantizationFunc.apply

    def forward(self, input):
        with torch.no_grad():
            self.scale = 1.4*torch.abs(input).mean()
        output = self.quantize_func(input, self.scale)
        return output

# 将自定义函数注册到 PyTorch 中
ternaryquantfunc = TernaryQuantizationFunc.apply