import torch
import torch.nn.functional as F
import numpy as np

class BiasCompensation(torch.nn.Module):
    def __init__(self,
                 channel_size, # output channel
                 channel_axis=(1,), # default feature_axis = 1
                # feature_shape=(1,1), # hxw
                 act_func=None,
                 clone_output=False,
                 ):
        super().__init__()
        self.channel_size = channel_size
        assert isinstance(channel_axis,(int,list,tuple)), f"channel_axis {channel_axis} is not int or list or tuple"
        self.channel_axis = channel_axis if isinstance(channel_axis,(list,tuple)) else (channel_axis,)
        self.bias_shape = None
        self.mean_axis = None
        self.bias = torch.nn.Parameter(torch.zeros(self.channel_size))
        # 假设有一个缩放scale和一个偏置bias的矩阵，用于纠正输出的误差
        # self.scale = nn.Parameter(torch.ones(self.scale_shape))
        self.nsamples = 0
        self.act_func = act_func
        self.clone_output = clone_output

    @torch.no_grad()
    def compute_shape(self,output):
        # compute feature_shape
        bias_shape=[]
        reduce_shape=[]
        self.channel_axis=tuple(x if x>=0 else output.dim()+x for x in self.channel_axis)
        
        for i in range(output.dim()):
            if i in self.channel_axis:
                bias_shape.append(output.size(i))
            else:
                bias_shape.append(1)
                reduce_shape.append(i)
        assert np.prod(bias_shape)==self.channel_size, f"bias_shape {bias_shape} is not compatible with channel_size {self.channel_size}"
        self.bias_shape = tuple(bias_shape)
        self.mean_axis = tuple(reduce_shape)

    def forward(self,output):
        # reshape
        if self.bias_shape is None:
            self.compute_shape(output)
        bias = self.bias.view(self.bias_shape)
        # bias = bias.to(output.data.dtype)
        if self.clone_output:
            output = output.clone()
        output += bias
        if self.act_func is not None:
            output = self.act_func(output)
        return output
        

    def update(self,float_output, quantized_output, post_bias=True):
        assert float_output.shape==quantized_output.shape, f"float_output.shape {float_output.shape} is not compatible with quantized_output.shape {quantized_output.shape}"
        if self.bias_shape is None:
            self.compute_shape(float_output)
        if post_bias:
            # post_bias=True: 
            # 表示这里的quantized_output是quantized_output+bias（post quantized）得到的，
            # 那更新的时候减去一个bias就得到以前的quantized_output了
            bias_update=float_output-quantized_output+self.bias.view(self.bias_shape).to(float_output.data.dtype)
            if len(self.mean_axis)>0:
                bias_update = torch.mean(bias_update,self.mean_axis)
        else:
            bias_update = float_output-quantized_output
            if len(self.mean_axis)>0:
                bias_update = torch.mean(bias_update,self.mean_axis)
        self.nsamples+=1   
        new_bias = (self.bias*(self.nsamples-1)+bias_update.view(-1))/self.nsamples
        self.bias.data.copy_(new_bias)