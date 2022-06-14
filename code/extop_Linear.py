import torch
from torch.autograd import gradcheck
import torch.nn as nn

class LinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx,input,weight,bias=None):
        #input:20,20
        #weight:30,20
        ctx.save_for_backward(input,weight,bias)
        output = input.mm(weight.t()) #20,20 x 20,30 -> 20,30
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx,grad_output):
        input,weight,bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input,grad_weight,grad_bias


    @staticmethod
    def symbolic(g,input,weight,bias):
        return g.op("LinearFunction",input,weight,bias)


class Linear(nn.Module):
    def __init__(self,input_features,output_features,bias=True):
        super(Linear,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.empty(output_features,input_features,dtype=torch.double))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features,dtype=torch.double))
        else:
            self.register_parameter("bias",None)
        
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)
    
    def forward(self,input):
        return LinearFunction.apply(input,self.weight, self.bias)


linear = LinearFunction.apply
input = torch.randn(20,20,dtype=torch.double,requires_grad=True)
weight = torch.randn(30,20,dtype=torch.double,requires_grad=True)
bias = torch.randn(30,dtype=torch.double,requires_grad=True)
inputs = (input, weight, bias)
test = gradcheck(linear,inputs,eps=1e-6,atol=1e-4)
print(test)
print('='*100)

model = Linear(20,30)
output = model(input)
print(output.shape)
torch.onnx.export(model,input,"model_8888.onnx")
