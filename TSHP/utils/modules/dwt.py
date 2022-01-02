# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

# DWT code borrow from https://github.com/LiQiufu/WaveSNet/blob/12cb9d24208c3d26917bf953618c30f0c6b0f03d/DWT_IDWT/DWT_IDWT_layer.py
import math
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class DWTFunction_1D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low, matrix_High):
        ctx.save_for_backward(matrix_Low, matrix_High)
        L = torch.matmul(input, matrix_Low.t()) # [B, C, L] @ [L, L//2] -> [B, 2*C, L//2]
        H = torch.matmul(input, matrix_High.t())# [B, C, L] @ [L, L//2] -> [B, 2*C, L//2]
        return L, H
    @staticmethod
    def backward(ctx, grad_L, grad_H):
        matrix_L, matrix_H = ctx.saved_variables
        grad_input = torch.add(torch.matmul(grad_L, matrix_L), torch.matmul(grad_H, matrix_H))
        return grad_input, None, None


class IDWTFunction_1D(Function):
    @staticmethod
    def forward(ctx, input_L, input_H, matrix_L, matrix_H):
        ctx.save_for_backward(matrix_L, matrix_H)
        output = torch.add(torch.matmul(input_L, matrix_L), torch.matmul(input_H, matrix_H))
        return output
    @staticmethod
    def backward(ctx, grad_output):
        matrix_L, matrix_H = ctx.saved_variables
        grad_L = torch.matmul(grad_output, matrix_L.t())
        grad_H = torch.matmul(grad_output, matrix_H.t())
        return grad_L, grad_H, None, None


__all__ = ['DWT_1D', 'IDWT_1D',]
class DWT_1D(nn.Module):
    """
    input: (N, C, L)
    output: L -- (N, C, L/2)
            H -- (N, C, L/2)
    """
    def __init__(self, wavename='db1'):
        """
        :param band_low: 小波分解所用低频滤波器组
        :param band_high: 小波分解所用高频滤波器组
        """
        super(DWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low  = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        self.matrix_ilen = None
        self.matrix_low  = None
        self.matrix_high = None
    
    def get_matrix(self, input):
        """
        生成变换矩阵
        :return:
        """
        L1 = input.shape[-1]
        if self.matrix_ilen is not None and self.matrix_ilen == L1:
            return# if the matrix for this input size is already calculated, do nothing
        assert L1 > self.band_length
        L = L1//2
        matrix_h = torch.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = torch.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:,(self.band_length_half-1):end]
        matrix_g = matrix_g[:,(self.band_length_half-1):end]
        self.matrix_ilen = L1
        self.matrix_low  = torch.tensor(matrix_h)
        self.matrix_high = torch.tensor(matrix_g)
        if input is not None:
            self.matrix_low  = self.matrix_low .to(input)
            self.matrix_high = self.matrix_high.to(input)
    
    def forward(self, input):
        assert input.dim() == 3
        self.get_matrix(input)
        return DWTFunction_1D.apply(input, self.matrix_low.to(input), self.matrix_high.to(input))


class IDWT_1D(nn.Module):
    """
    input:  L -- (N, C, L/2)
            H -- (N, C, L/2)
    output: (N, C, L)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波重建所需低频滤波器组
        :param band_high: 小波重建所需高频滤波器组
        """
        super(IDWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = torch.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = torch.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:,(self.band_length_half-1):end]
        matrix_g = matrix_g[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.tensor(matrix_h).cuda()
            self.matrix_high = torch.tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.tensor(matrix_h)
            self.matrix_high = torch.tensor(matrix_g)

    def forward(self, L, H):
        assert len(L.size()) == len(H.size()) == 3
        self.input_height = L.size()[-1] + H.size()[-1]
        #assert self.input_height > self.band_length
        self.get_matrix()
        return IDWTFunction_1D.apply(L, H, self.matrix_low, self.matrix_high)