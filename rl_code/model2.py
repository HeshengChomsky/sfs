# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from helpers import SinusoidalPosEmb

class Mish(keras.Model):
    def __init__(self):
        super(Mish, self).__init__()
    def call(self, inputs, training=None, mask=None):
        x=inputs*(tf.tanh(tf.nn.leaky_relu(inputs)))
        return x

class MLP(keras.Model):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 t_dim=16):

        super(MLP, self).__init__()
        self.time_mlp =keras.Sequential([
            SinusoidalPosEmb(t_dim),
            layers.Dense(t_dim*2),
            Mish(),
            layers.Dense(t_dim)
        ])
        # self.time_mlp = nn.Sequential(
        #     SinusoidalPosEmb(t_dim),
        #     nn.Linear(t_dim, t_dim * 2),
        #     nn.Mish(),
        #     nn.Linear(t_dim * 2, t_dim),
        # )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer=keras.Sequential([
            layers.Dense(128),
            Mish(),
            layers.Dense(256),
            Mish(),
            layers.Dense(256),
            Mish()
        ])
        # self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
        #                                nn.Mish(),
        #                                nn.Linear(256, 256),
        #                                nn.Mish(),
        #                                nn.Linear(256, 256),
        #                                nn.Mish())

        # self.final_layer = nn.Linear(256, action_dim)
        self.final_layer=layers.Dense(action_dim)
    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = tf.concat([x, t, state], axis=1)
        x = self.mid_layer(x)

        return self.final_layer(x)

    def call(self, x, time, state, training=None, mask=None):
        time=tf.cast(time,dtype=tf.float32)

        return self.forward(x,time,state)

if __name__ == '__main__':
    c=MLP(state_dim=32,action_dim=20)
    state=tf.random_normal([64,32])
    x=tf.random_normal([64,20])
    t=tf.fill([64,],2)
    temp=c.forward(x,t,state)
    print(temp.shape)

