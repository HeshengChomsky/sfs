# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import math
import time
import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
class SinusoidalPosEmb(keras.Model):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim


    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = tf.exp(tf.cast(keras.backend.arange(half_dim),dtype=tf.float32) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

    def call(self, inputs, training=None, mask=None):
        inputs=tf.cast(inputs,dtype=tf.float32)
        return self.forward(inputs)

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#


def extract(a, t, x_shape):
    b, *_ = t.shape
    out=tf.gather(a,t,axis=-1)
    # out = a.gather(-1, t)
    return tf.reshape(out,[b,*((1,) * (len(x_shape) - 1))])
    # return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=tf.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return tf.convert_to_tensor(betas_clipped,dtype=dtype)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=tf.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )

    return tf.convert_to_tensor(betas,dtype=dtype)


def vp_beta_schedule(timesteps, dtype=tf.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return tf.convert_to_tensor(betas, dtype=dtype)

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(keras.Model):

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        '''
            pred, targ : tensor [ batch_size x action_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = tf.reduce_mean(loss * weights)
        return weighted_loss
    def call(self, pred,targ, training=None, mask=None):
        return self.forward(pred,targ)

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return tf.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return tf.reduce_mean(keras.losses.MSE(pred, targ))

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
}


class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

if __name__ == '__main__':
    s=SinusoidalPosEmb(16)
    t=tf.fill([64,],2)
    temp=s(t)
    print(temp.shape)