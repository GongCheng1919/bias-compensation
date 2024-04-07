import torch
import torch.nn.functional as F
import numpy as np

class UL2PTQ(torch.nn.Module):
    def __init__(self,
                 channel_size, # output channel
                 feature_size, # hxw
                 LoRA_dim=None,
                 static_mode=True):
        # static_mode: 维护一个静态的偏置矩阵，使其可以为每一层的输出添加一个补偿矩阵，从而降低量化误差，提升精度
        # dynamic_mode (static_mode=False): 从输入中动态的计算偏置矩阵，使其可以为每一层的输出添加一个补偿矩阵，从而降低量化误差，提升精度
        super().__init__()
        self.static_mode = static_mode
        self.channel_size = channel_size
        self.feature_size = feature_size
        self.bias_shape = (1,self.channel_size)+self.feature_size
        # 默认使用1/4的秩以降低计算量和内存占用
        self.bias = torch.nn.Parameter(torch.zeros(self.bias_shape))
        self.nsamples = 0

    def forward(self,input,output):
        if self.static_mode:
            bias = self.bias
        else:
            # 如果通道数不同，那可能需要使用卷积层，为此我们可以使用1x1卷积层来实现
            # 我们为了降低计算量，可以使用固定的卷积层来实现，比如全为1的卷积核，或者高斯卷积核
            with torch.no_grad():
                if isinstance(input,(list,tuple)):
                    input = input[0]
                channel_map = torch.ones(output.size(1),input.size(1),1,1).to(input.device)
                trans_map = F.conv2d(input,weight=channel_map)
                trans_map = F.interpolate(trans_map, size=self.feature_size, mode='bilinear', align_corners=False)
            bias = self.bias*trans_map
        if self.feature_size[0]!=output.size(2) or self.feature_size[1]!=output.size(3):
            bias = F.interpolate(bias, size=output.shape[2:], mode='bilinear', align_corners=False)
        
        return output + bias

class UL2PTQ2(torch.nn.Module):
    def __init__(self,
                 channel_size, # output channel
                 feature_size, # hxw
                 ):
        super().__init__()
        self.channel_size = channel_size
        self.feature_size = feature_size
        self.bias_shape = (1,self.channel_size)+self.feature_size
        # self.scale_shape = (1,self.channel_size,1,1) # 逐通道缩放
        # 默认使用1/4的秩以降低计算量和内存占用
        self.register_buffer('bias', (torch.zeros(self.bias_shape)))
        # 假设有一个缩放scale和一个偏置bias的矩阵，用于纠正输出的误差
        # self.register_buffer('scale', (torch.ones(self.scale_shape)))

    def forward(self,output):
        # if self.feature_size[0]!=output.size(2) or self.feature_size[1]!=output.size(3):
        bias = F.interpolate(self.bias, size=output.shape[2:], mode='bilinear', align_corners=False)
        return output + bias
    
    def update(self,float_output, quantized_output, post_bias=True):
        # 根据float_out和quantized_out来更新bias
        # 现在的output为quantized_out，不包含post_quant的结果
        # udpate bias: B = mean(O_f-s*O_q)
        bias = F.interpolate(self.bias, size=float_output.shape[2:], mode='bilinear', align_corners=False)
        if post_bias:
            bias_update = torch.mean(float_output-self.scale*(quantized_output-bias),0).unsqueeze(0)
        else:
            bias_update = torch.mean(float_output-self.scale*quantized_output,0).unsqueeze(0)
        new_bias = (bias+bias_update)/2.0
        # if self.feature_size[0]!=float_output.size(2) or self.feature_size[1]!=float_output.size(3):
        new_bias = F.interpolate(new_bias, size=self.feature_size, mode='bilinear', align_corners=False)

        # update scale: mean((O_f-B)/O_q)
        # new_scale = torch.sum(float_output-bias,[0,2,3])/torch.sum(quantized_output,[0,2,3])
        # new_scale = new_scale.reshape(self.scale_shape)
        # new_scale = torch.clamp(new_scale,0.5,2.0) # 限制其范围，不然会剧烈变化
        # print(new_scale.shape)
        
        self.bias.copy_(new_bias)
        # self.scale.copy_(new_scale)

class UL2PTQ3(torch.nn.Module):
    __layer_type__ = ['conv','linear']
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
        # if layer_type in self.__layer_type__:
        #     self.layer_type = layer_type
        # if self.layer_type=='conv':
        #     self.bias_shape = (1,self.channel_size,1,1) # 逐通道缩放
        #     self.mean_axis = (0,2,3)
        # elif self.layer_type=='linear':
        #     self.bias_shape = (1,self.channel_size)
        #     self.mean_axis = (0,)
        # self.scale_shape = (1,self.channel_size,1,1) # 逐通道缩放
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