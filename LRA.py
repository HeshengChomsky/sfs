import math
import numpy as np
import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Normal
# from torch.distributions.transformed_distribution import TransformedDistribution
# from torch.distributions.transforms import TanhTransform

import tensorflow as tf
from tensorflow import keras



MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-7


class Guide_policy(keras.Model):
    def __init__(self, state_dim, log_std_multiplier=1.0, log_std_offset=-1.0, hidden_num=256):
        super(Guide_policy, self).__init__()

        self.fc1 = keras.layers.Dense(hidden_num)
        self.fc2 = keras.layers.Dense(hidden_num)
        self.mu_head = keras.layers.Dense(state_dim)
        self.sigma_head = keras.layers.Dense(state_dim)

        self.log_sigma_multiplier = log_std_multiplier
        self.log_sigma_offset = log_std_offset

    def _get_outputs(self, state,hguide):
        cont_state=tf.concat([state,hguide],axis=1)
        a = tf.nn.relu(self.fc1(cont_state))
        a = tf.nn.relu(self.fc2(a))
        mu = self.mu_head(a)
        mu=tf.clip_by_value(mu,MEAN_MIN,MEAN_MAX)
        # mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        # log_sigma = self.log_sigma_multiplier * log_sigma + self.log_sigma_offset

        # log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        log_sigma=tf.clip_by_value(log_sigma,LOG_STD_MIN,LOG_STD_MAX)
        sigma = tf.exp(log_sigma)

        a_distribution = tf.compat.v1.distributions.Normal(mu,sigma)

        # a_distribution = TransformedDistribution(
        #     Normal(mu, sigma), TanhTransform(cache_size=1)
        # )
        a_tanh_mode = tf.tanh(mu)
        return a_distribution, a_tanh_mode

    def forward(self, state,hguide):
        a_dist, a_tanh_mode = self._get_outputs(state,hguide)
        action = a_dist.sample()
        logp_pi = tf.reduce_sum(a_dist.log_prob(action),axis=-1)
        return action, logp_pi, a_tanh_mode

    def call(self, state,hguide, training=None, mask=None):
        return self.forward(state,hguide)

    def get_log_density(self, state,hguide, action):
        a_dist, _ = self._get_outputs(state,hguide)
        # action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        action_clip=tf.clip_by_value(action, -1. + EPS, 1. - EPS)
        logp_action = a_dist.log_prob(action_clip)
        return logp_action


class Execute_policy(keras.Model):
    def __init__(self, state_dim, action_dim, log_std_multiplier=1.0, log_std_offset=-1.0, hidden_num=512):
        super(Execute_policy, self).__init__()

        self.fc1 = keras.layers.Dense(hidden_num)
        self.fc2 = keras.layers.Dense(hidden_num)
        self.mu_head = keras.layers.Dense(action_dim)
        self.sigma_head = keras.layers.Dense(action_dim)

        self.log_sigma_multiplier = log_std_multiplier
        self.log_sigma_offset = log_std_offset

    def _get_outputs(self, state, goal):
        concat_state = tf.concat([state, goal], axis=1)
        a = tf.nn.relu(self.fc1(concat_state))
        a = tf.nn.relu(self.fc2(a))

        mu = self.mu_head(a)
        mu=tf.clip_by_value(mu, MEAN_MIN, MEAN_MAX)
        # mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        # log_sigma = self.log_sigma_multiplier * log_sigma + self.log_sigma_offset
        log_sigma=tf.clip_by_value(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        # log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = tf.exp(log_sigma)


        a_distribution=tf.compat.v1.distributions.Normal(mu,sigma)
        # a_distribution = TransformedDistribution(
        #     Normal(mu, sigma), TanhTransform(cache_size=1)
        # )
        a_tanh_mode=tf.tanh(mu)
        # a_tanh_mode = torch.tanh(mu)
        return a_distribution, a_tanh_mode

    def forward(self, state, goal):
        a_dist, a_tanh_mode = self._get_outputs(state, goal)
        action = a_dist.sample()
        logp_pi = tf.reduce_sum(a_dist.log_prob(action),axis=-1)
        return action, logp_pi, a_tanh_mode

    def call(self, state,goal, training=None, mask=None):
        return self.forward(state,goal)

    def get_log_density(self, state, goal, action):
        a_dist, _ = self._get_outputs(state, goal)
        action_clip=tf.clip_by_value(action, -1. + EPS, 1. - EPS)
        # action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_action = a_dist.log_prob(action_clip)
        return logp_action


class Double_Critic(keras.Model):
    def __init__(self, state_dim):
        super(Double_Critic, self).__init__()

        # V1 architecture
        self.l1 = keras.layers.Dense(256)
        self.l2 = keras.layers.Dense(256)
        self.l3 = keras.layers.Dense(1)

        # V2 architecture
        self.l4 = keras.layers.Dense(256)
        self.l5 = keras.layers.Dense(256)
        self.l6 = keras.layers.Dense(1)

    def forward(self, state):
        v1 = tf.nn.relu(self.l1(state))
        v1 = tf.nn.relu(self.l2(v1))
        v1 = self.l3(v1)

        v2 = tf.nn.relu(self.l4(state))
        v2 = tf.nn.relu(self.l5(v2))
        v2 = self.l6(v2)
        return v1, v2

    def V1(self, state):
        v1 = tf.nn.relu(self.l1(state))
        v1 = tf.nn.relu(self.l2(v1))
        v1 = self.l3(v1)
        return v1
    def call(self, state, training=None, mask=None):
        return self.forward(state)


def loss(diff, expectile=0.8):
    weight =tf.where(diff>0,tf.zeros_like(diff)+expectile,tf.ones_like(diff)-expectile)
    # weight = tf.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class POR(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            eta=0.005,
            tau=0.9,
            alpha=10.0,
            lmbda=10.0,
            g_v=False,
            e_weight=True,
    ):

        self.policy_e = Execute_policy(state_dim, action_dim)
        self.policy_e_optimizer = keras.optimizers.Adam(lr=3e-4)

        self.policy_g = Guide_policy(state_dim)
        self.policy_g_optimizer = keras.optimizers.Adam(lr=3e-4)

        self.critic = Double_Critic(state_dim)
        self.critic_target = Double_Critic(state_dim)
        self.critic_optimizer = keras.optimizers.Adam(lr=3e-4)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.tau = tau
        self.alpha = alpha
        self.lmbda = lmbda
        self.g_v = g_v
        self.e_weight = e_weight

        self.discount = discount
        self.eta = eta
        self.total_it = 0

    def select_action(self, state,action_h):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        _, _, goal = self.policy_g(state,action_h)
        _, _, action = self.policy_e(state, goal)
        return action

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        # state, action, next_state, reward, not_done,budget = replay_buffer.sample(batch_size)
        state = tf.random_normal([64, 32])
        action = tf.random_normal([64, 20])
        next_state = tf.random_normal([64, 32])
        reward = tf.random_normal([64,1])
        not_done = tf.zeros_like(reward)
        budget=tf.random_normal([64,1])
        # Update V
        # with torch.no_grad():
        next_v1, next_v2 = self.critic_target(next_state)
        next_v = tf.minimum(next_v1, next_v2)
        # next_v = next_v1
        target_v = (reward + self.discount * (1-not_done) * next_v)
        with tf.GradientTape() as tape:
            v1, v2 = self.critic(state)

            critic_loss =  tf.reduce_mean(loss(target_v - v1, self.tau) + loss(target_v - v2, self.tau))
        # critic_loss = loss(target_v - v1, self.tau).mean()
        grads=tape.gradient(critic_loss,self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads,self.critic.trainable_variables))
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()

        # Update guide-policy
        # with torch.no_grad():
        next_v1, next_v2 = self.critic_target(next_state)
        next_v = tf.minimum(next_v1, next_v2)
        target_v = (reward + self.discount * not_done * next_v)
        v1, v2 = self.critic(state)
        residual = target_v - v1
        weight = tf.exp(residual * self.alpha)
        weight=tf.squeeze(tf.clip_by_value(weight,-100.0,100.0),axis=-1)
        # weight = torch.clamp(weight, max=100.0).squeeze(-1).detach()
        with tf.GradientTape() as tape:
            log_pi_g = self.policy_g.get_log_density(state,budget, next_state)
            log_pi_g=tf.reduce_sum(log_pi_g,axis=1)
            if not self.g_v:
                p_g_loss = -tf.reduce_mean(weight * log_pi_g)
            else:
                g, _, _ = self.policy_g(state)
                v1_g, v2_g = self.critic(g)
                min_v_g = tf.squeeze(tf.minimum(v1_g, v2_g))
                lmbda = self.lmbda / tf.reduce_mean(min_v_g.abs())
                p_g_loss = -tf.reduce_mean(weight * log_pi_g + lmbda * min_v_g)

        grads_pg=tape.gradient(p_g_loss,self.policy_g.trainable_variables)
        self.policy_g_optimizer.apply_gradients(zip(grads_pg,self.policy_g.trainable_variables))

        # self.policy_g_optimizer.zero_grad()
        # p_g_loss.backward()
        # self.policy_g_optimizer.step()

        # Update execute-policy
        with tf.GradientTape() as tape:
            log_pi_a = self.policy_e.get_log_density(state, next_state, action)
            log_pi_a = tf.reduce_sum(log_pi_a, axis=1)
            if self.e_weight:
                p_e_loss = -tf.reduce_mean(weight * log_pi_a)
            else:
                p_e_loss = -tf.reduce_mean(log_pi_a)

        grads_pe=tape.gradient(p_e_loss,self.policy_e.trainable_variables)
        self.policy_e_optimizer.apply_gradients(zip(grads_pe,self.policy_e.trainable_variables))

        # self.policy_e_optimizer.zero_grad()
        # p_e_loss.backward()
        # self.policy_e_optimizer.step()

        if self.total_it % 50 == 0:
            print(f'mean target v value is {tf.reduce_mean(target_v)}')
            print(f'mean v1 value is {tf.reduce_mean(v1)}')
            print(f'mean residual is {tf.reduce_mean(residual)}')

        # Update the frozen target models
        # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        #     target_param.data.copy_(self.eta * param.data + (1 - self.eta) * target_param.data)
        for source_weight, target_weight in zip(self.critic.trainable_variables,
                                                self.critic_target.trainable_variables):
            target_weight.assign(self.tau * source_weight + (1.0 - self.tau) * target_weight)

    def save(self, filename):
        self.policy_g.save_weights(filename + "_policy_g")
        self.policy_e.save_weights(filename + "_policy_e")
        # torch.save(self.policy_g.state_dict(), filename + "_policy_g")
        # torch.save(self.policy_g_optimizer.state_dict(), filename + "_policy_g_optimizer")
        # torch.save(self.policy_e.state_dict(), filename + "_policy_e")
        # torch.save(self.policy_e_optimizer.state_dict(), filename + "_policy_e_optimizer")

    def load(self, filename):
        self.policy_g.load_weights(filename + "_policy_g")
        self.policy_e.load_weights(filename + "_policy_e")
        # self.policy_g.load_state_dict(torch.load(filename + "_policy_g"))
        # self.policy_g_optimizer.load_state_dict(torch.load(filename + "_policy_g_optimizer"))
        # self.policy_e.load_state_dict(torch.load(filename + "_policy_e"))
        # self.policy_e_optimizer.load_state_dict(torch.load(filename + "_policy_e_optimizer"))

if __name__ == '__main__':
    agent=POR(32,20,1)
    state=tf.random_normal([64,32])
    action_h=tf.random_normal([64,4])
    tt=agent.train(replay_buffer=None,batch_size=64)
    # action=agent.select_action(state,action_h)
    # print(action.shape)