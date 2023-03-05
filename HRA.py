# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from myutils.logger import logger
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from diffusion import Diffusion
from model2 import MLP
from helpers import EMA


class Critic(keras.Model):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model=keras.Sequential([layers.Dense(hidden_dim,activation=tf.nn.relu),
                                   layers.Dense(hidden_dim,activation=tf.nn.relu),
                                    layers.Dense(hidden_dim, activation=tf.nn.relu),
                                    layers.Dense(1, activation=tf.nn.relu)
                                    ])
        self.q2_model = keras.Sequential([layers.Dense(hidden_dim, activation=tf.nn.relu),
                                          layers.Dense(hidden_dim, activation=tf.nn.relu),
                                          layers.Dense(hidden_dim, activation=tf.nn.relu),
                                          layers.Dense(1, activation=tf.nn.relu)
                                          ])
        # self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
        #                               nn.Mish(),
        #                               nn.Linear(hidden_dim, hidden_dim),
        #                               nn.Mish(),
        #                               nn.Linear(hidden_dim, hidden_dim),
        #                               nn.Mish(),
        #                               nn.Linear(hidden_dim, 1))

        # self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
        #                               nn.Mish(),
        #                               nn.Linear(hidden_dim, hidden_dim),
        #                               nn.Mish(),
        #                               nn.Linear(hidden_dim, hidden_dim),
        #                               nn.Mish(),
        #                               nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = tf.concat([state, action], axis=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = tf.concat([state, action], axis=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return tf.minimum(q1, q2)

    def call(self, inputs,action,training=None, mask=None):
        return self.forward(inputs,action)


class Diffusion_QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.actor_optimizer=keras.optimizers.Adam(learning_rate=lr)


        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.critic_optimizer=keras.optimizers.Adam(lr=3e-4)

        # if lr_decay:
        #     self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
        #     self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.max_q_backup = max_q_backup

    def step_ema(self):
        if self.step < self.step_start_ema:
            return self.ema_model.set_weights(self.actor.get_weights())
        # self.ema.update_model_average(self.ema_model, self.actor)


    def sample_action(self, state):
        # state=tf.convert_to_tensor(tf.reshape(state,[1,-1]),dtype=tf.float32)
        state_rpt=tf.repeat(state,repeats=50,axis=0)
        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        action = self.actor.sample(state_rpt)
        q_value = self.critic_target.q_min(state_rpt, action)
        q_value=tf.reshape(q_value,[1,-1])
        idx = tf.random.categorical(tf.nn.softmax(q_value,axis=1), 1)
        return action[idx[0,0]]

    def save_model(self, dir, id=None):
        if id is not None:
            self.actor.save_weights(f'{dir}/actor_{id}.pth')
            self.critic.save_weights(f'{dir}/critic_{id}.pth')
            # torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            # torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            self.actor.save_weights(f'{dir}/actor.pth')
            self.critic.save_weights(f'{dir}/critic.pth')
            # torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            # torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_weights(f'{dir}/actor_{id}.pth')
            self.critic.load_weights(f'{dir}/critic_{id}.pth')
            # self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            # self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_weights(f'{dir}/actor.pth')
            self.critic.load_weights(f'{dir}/critic.pth')
            # self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            # self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            # state, action, next_state, reward, not_done, budget, aimcpc = replay_buffer.sample(batch_size)
            state = tf.random_normal([64, 32])
            action = tf.random_normal([64, 20])
            next_state = tf.random_normal([64, 32])
            reward = tf.random_normal([64,1])
            not_done = tf.zeros_like(reward)

            """ Q Training """
            with tf.GradientTape() as tape:
                current_q1, current_q2 = self.critic(state, action)
                if self.max_q_backup:
                    next_state_rpt = tf.repeat(next_state, repeats=10, axis=0)
                    # next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                    next_action_rpt = self.ema_model(next_state_rpt)
                    target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                    target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                    target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                    target_q = tf.minimum(target_q1, target_q2)
                else:
                    next_action = self.ema_model(next_state)
                    target_q1, target_q2 = self.critic_target(next_state, next_action)
                    target_q = tf.minimum(target_q1, target_q2)

                target_q = (reward + (1 - not_done) * self.discount * target_q)
                critic_loss = keras.losses.MSE(target_q,current_q1) + keras.losses.MSE(target_q,current_q2)
                critic_loss=tf.reduce_mean(critic_loss)
                # session=tf.Session()
                # session.run(tf.Print(critic_loss,[critic_loss]))
                # session.close()
                # session=tf.InteractiveSession()
                # session.run(tf.Print(critic_loss,[critic_loss]))
                # session.close()
            grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

            # self.critic_optimizer.zero_grad()
            # critic_loss.backward()
            # if self.grad_norm > 0:
            #     critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm,
            #                                                  norm_type=2)
            # self.critic_optimizer.step()

            """ Policy Training """
            with tf.GradientTape() as tape:
                bc_loss = self.actor.loss(action, state)
                new_action = self.actor(state)

                q1_new_action, q2_new_action = self.critic(state, new_action)
                if np.random.uniform() > 0.5:
                    q_loss = - tf.reduce_mean(q1_new_action) / tf.reduce_mean(tf.abs(q2_new_action))
                else:
                    q_loss = - tf.reduce_mean(q2_new_action) / tf.reduce_mean(tf.abs(q1_new_action))
                actor_loss = bc_loss + self.eta * q_loss

            grads_actor = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))

            # self.actor_optimizer.zero_grad()
            # actor_loss.backward()
            # if self.grad_norm > 0:
            #     actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm,
            #                                                 norm_type=2)
            # self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # for Q,Q_target in zip(self.critic.trainable_variables,self.critic_target.trainable_variables):
            for source_weight, target_weight in zip(self.critic.trainable_variables, self.critic_target.trainable_variables):
                target_weight.assign(self.tau * source_weight + (1.0 - self.tau) * target_weight)

            self.step += 1

            # """ Log """
            # if log_writer is not None:
            #     if self.grad_norm > 0:
            #         log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
            #         log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            #     log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
            #     log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
            #     log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
            #     log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)
            metric['actor_loss'].append(actor_loss)
            metric['bc_loss'].append(bc_loss)
            metric['ql_loss'].append(q_loss)
            metric['critic_loss'].append(critic_loss)

        # if self.lr_decay:
        #     self.actor_lr_scheduler.step()
        #     self.critic_lr_scheduler.step()

        return metric

if __name__ == '__main__':
    diffusion=Diffusion_QL(state_dim=32,action_dim=20,max_action=1.,discount=0.98,tau=0.2)
    state=tf.random_normal([1,32])
    metric=diffusion.train(replay_buffer=None,iterations=1,batch_size=64)
    # action=diffusion.sample_action(state)

