import torch

class QuantizationFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, max_val, min_val):
        # 前向传播函数
        # quantize
        # print("Apply quantization in Forward")
        output = (input-zero_point) / scale 
        # print("forward output=",output,"max_val=",max_val,"min_val=",min_val)
        clamp_index = None
        if input.requires_grad:
            clamp_index = (output < min_val) | (output > max_val)
        ctx.save_for_backward(clamp_index)
        # print("forward clamp_index=",clamp_index)
        output = torch.clamp(output,min=min_val, max=max_val)
        # print(input.max(),input.min())
        # print(output.max(),output.min())
        output = torch.round(output)
        # dequantize
        output = output * scale+zero_point
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播函数
        # print("Apply quantization in Backward")
        clamp_index, = ctx.saved_tensors
        # print("backward clamp_index=",clamp_index)
        grad_input = None
        if clamp_index is not None:
            grad_input = grad_output.clone()
            grad_input[clamp_index] = 0
        return grad_input, None, None, None, None

class FixedPointQunatization(torch.nn.Module):
    def __init__(self,bitwidth,asymmtric=True):
        super().__init__()
        self.asymmtric = asymmtric
        self.bitwidth = bitwidth
        if self.asymmtric:
            self.max_val = 2**(bitwidth-1)-1# 3 std
            self.min_val = -2**(bitwidth-1)+1# -3 std
        else:
            self.max_val = 2**(bitwidth)-1# 3 std
            self.min_val = 0
        self.quantize_func = QuantizationFunc.apply
        self.clip_rate = 1

    def forward(self, input):
        if self.asymmtric:
            zero_point = input.mean()
            scale = (3*input.std())/(self.max_val*self.clip_rate)
        else:
            zero_point = input.min() 
            scale = (input.max()-zero_point)/(self.max_val*self.clip_rate)
        output = self.quantize_func(input, scale, zero_point, self.max_val, self.min_val)
        return output

class postFixedQ(torch.nn.Module):
    def __init__(self,bitwidth=4,asymmtric=True):
        super().__init__()
        self.fixedQ = FixedPointQunatization(bitwidth=4,asymmtric=True)
    def forward(self,output):
        return self.fixedQ(output)

# 将自定义函数注册到 PyTorch 中
linearquantfunc = QuantizationFunc.apply