# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras

from helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            Losses)
from utils import Progress, Silent


class Diffusion(keras.Model):
    def __init__(self, state_dim, action_dim, model, max_action,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod=tf.cumprod(alphas,axis=0)
        alphas_cumprod_prev=tf.concat([tf.ones([1]),alphas_cumprod[:-1]],axis=0)
        # alphas_cumprod = torch.cumprod(alphas, axis=0)
        # alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.betas=betas
        self.alphas_cumprod=alphas_cumprod
        self.alphas_cumprod_prev=alphas_cumprod_prev
        self.sqrt_alphas_cumprod=tf.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod=tf.sqrt(1. - alphas_cumprod)
        self.log_one_minus_alphas_cumprod=tf.log(1. - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod=tf.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod=tf.sqrt(1. / alphas_cumprod - 1)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance=posterior_variance
        self.posterior_log_variance_clipped=tf.log(tf.clip_by_value(posterior_variance,clip_value_min=1e-20,clip_value_max=1e20))
        self.posterior_mean_coef1=betas * tf.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2=(1. - alphas_cumprod_prev) * tf.sqrt(alphas) / (1. - alphas_cumprod)

        # self.register_buffer('betas', betas)
        # self.register_buffer('alphas_cumprod', alphas_cumprod)
        # self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        #
        # # calculations for diffusion q(x_t | x_{t-1}) and others
        # self.register_buffer('sqrt_alphas_cumprod', tf.sqrt(alphas_cumprod))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', tf.sqrt(1. - alphas_cumprod))
        # self.register_buffer('log_one_minus_alphas_cumprod', tf.log(1. - alphas_cumprod))
        # self.register_buffer('sqrt_recip_alphas_cumprod', tf.sqrt(1. / alphas_cumprod))
        # self.register_buffer('sqrt_recipm1_alphas_cumprod', tf.sqrt(1. / alphas_cumprod - 1))
        #
        # # calculations for posterior q(x_{t-1} | x_t, x_0)
        # posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # self.register_buffer('posterior_variance', posterior_variance)
        #
        # ## log calculation clipped because the posterior variance
        # ## is 0 at the beginning of the diffusion chain
        # self.register_buffer('posterior_log_variance_clipped',
        #                      tf.log(tf.clip_by_value(posterior_variance,clip_value_min=1e-20)))
        # self.register_buffer('posterior_mean_coef1',
        #                      betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        # self.register_buffer('posterior_mean_coef2',
        #                      (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon=tf.clip_by_value(x_recon,-self.max_action,self.max_action)
            # x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        # noise = torch.randn_like(x)
        noise=tf.random_normal(x.shape)

        # no noise when t == 0
        zose=tf.zeros_like(t)
        nonzero_mask = tf.reshape((1 - tf.cast(tf.equal(t,zose),dtype=tf.float32)),[b, *((1,) * (len(x.shape) - 1))])
        return model_mean + nonzero_mask *tf.exp(0.5 * model_log_variance)*noise
        # return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):

        batch_size = shape[0]
        # x = torch.randn(shape, device=device)
        x=tf.random_normal(shape)

        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            # timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            timesteps=tf.fill([batch_size,],i)

            x = self.p_sample(x, timesteps, state)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            # return x, torch.stack(diffusion, dim=1)
            return x,tf.stack(diffusion,axis=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return tf.clip_by_value(action,-self.max_action, self.max_action)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            # noise = torch.randn_like(x_start)
            noise =tf.random_normal(x_start.shape)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        # noise = torch.randn_like(x_start)
        noise=tf.random_normal(x_start.shape)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = x.shape[0]
        # t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        t=np.random.randint(0,self.n_timesteps,[batch_size,])
        t = tf.convert_to_tensor(t,dtype=tf.int64)
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)

