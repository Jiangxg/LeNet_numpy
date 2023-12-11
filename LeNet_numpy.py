import numpy as np
import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import math

def conv2d(kernel_weight, kernel_bias, image_slice):
    '''
    Args:
        kernel_weight: 1*1*kernel_size*kernel_size
        kernel_bias: 1*1*kernel_size*kernel_size
        image_slice: N*1*H*W
    '''
    kernel_size = kernel_weight.shape[-1]
    N, _, H_in, W_in = image_slice.shape
    H_out = H_in - kernel_size + 1
    W_out = W_in - kernel_size + 1
    output = np.zeros(N, 1, H_out, W_out)

    # loop for convolution
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                for k_i in range(kernel_size):
                    for k_j in range(kernel_size):
                        output[n, 0, i, j] += kernel_weight[0, 0, k_i, k_j] * \
                                    image_slice[n, 0, i + k_i, j + k_j] + kernel_bias[0, 0, k_i, k_j]
    return output
  
def maxpool(image_slice, kernel_size, stride):
    '''
    Args:
        image_slice: N*1*H*W
    '''
    N, _, H_in, W_in = image_slice.shape
    H_out = (H_in - kernel_size) / stride + 1
    W_out = (W_in - kernel_size) / stride + 1
    output = np.zeros(N, 1, H_out, W_out)

    # loop for convolution
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                temp = image_slice[n, 0, i*stride, j*stride]
                for k_i in range(kernel_size):
                    for k_j in range(kernel_size):
                        temp = max(image_slice[n, 0, i*stride + k_i, j*stride + k_j], temp)
                    output[n, 0, i, j] = temp
    return output


def param_init(shape, layer):
    assert layer == "conv" or layer == "linear", "argument layer must be conv or linear"
    if layer == "conv":
        k = 1. / (3 * shape[-1]* shape[-2])
    else:
        k = 1. / shape[-1]
    
    return np.random.uniform(-math.sqrt(k), math.sqrt(k),shape)

class My_Conv2d():
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size):
        '''
        shape of input array: N, C, H, W
        shape of input array: N, C', H', W'
        '''
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = param_init((out_channels, in_channels, kernel_size, kernel_size), layer="conv")
        self.bias = param_init((out_channels, in_channels, kernel_size, kernel_size), layer="conv")

    def __call__(self, input):
        return self.forward(input)


    def forward(self, input):   
        # input must with a shape of N*C*H*W
        assert len(input.shape) == 4, "input array must with a shape of N*C*H*W!"
        assert input.shape[1] == self.in_channels, "wrong number of in_channel!"

        N, C_in, H_in, W_in = input.shape
        H_out = H_in - self.kernel_size + 1
        W_out = W_in - self.kernel_size + 1

        # save input for backward
        self.input = input

        # conduct convolution
        output = np.zeros(N, self.out_channels, H_out, W_out)
        for cin in range(self.in_channels):
            for cout in range(self.out_channels):
                output[:, cout:cout+1, ...] += conv2d(self.weight[cout:cout+1, cin:cin+1, ...], 
                                        self.bias[cout:cout+1, cin:cin+1, ...], input[:, cin:cin+1, ...])
        return output

    def backward(self, grad_output):
        pass
        return grad_input, grad_weight, grad_bias


class My_MaxPool2d():
    def __init__(self,
            kernel_size=2):
        '''
        shape of input array: N, C, H, W
        shape of input array: N, C', H', W'
        '''
        self.kernel_size = kernel_size
        self.stride = kernel_size

    def __call__(self, input):
        return self.forward(input)


    def forward(self, input):   
        # input must with a shape of N*C*H*W
        assert len(input.shape) == 4, "input array must with a shape of N*C*H*W!"

        N, cin, H_in, W_in = input.shape
        H_out = (H_in - self.kernel_size) / self.stride + 1
        W_out = (W_in - self.kernel_size) / self.stride + 1

        # conduct max pooling
        output = np.zeros(N, cin, H_out, W_out)
        for cin in range(self.in_channels):
            output[:, cin:cin+1, ...] = maxpool(input[:, cin:cin+1, ...], self.kernel_size, self.stride)
        return output

    def backward(self, grad_output):
        pass
        return grad_input

class My_Flatten():
    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        # input: shape of N*C*H*W
        # output: shape of N*H_out
        # saved for backward
        assert len(input.shape) == 4, "the input of Flatten should have a shape of N*C*H*W"
        self.input_shape = input.shape
        return input.reshape(self.input_shape[0], -1)

    def backward(self, grad_output):
        grad_input = grad_output.reshape(self.input_shape)
        return grad_input

class My_Linear():
    def __init__(self, 
                in_features,
                out_features):
        self.in_features = in_features
        self.out_features = out_features

        # out_features, in_features
        self.weight = param_init((self.out_features, self.in_features), layer="linear")
        # out_features
        self.bias = param_init((self.out_features, self.in_features), layer="linear")[:, 0]

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        # self.input: shape of N*H_in
        assert len(input.shape) == 2, "the input of Linear should has a shape of N*H_in"
        self.input = input
        output = np.matmul(input, np.transpose(self.weight)) + self.bias
        return output

    def backward(self, grad_output):
        return grad_input, grad_weight, grad_bias
