import numpy as np
import os
import collections
import random
import tensorflow as tf

class HData_Sampler(object):
    def __init__(self, data):

        self.states=tf.convert_to_tensor(np.array(data['states']),dtype=tf.float32)
        self.rewards=tf.convert_to_tensor(np.array(data['rewards']),dtype=tf.float32)
        self.actions=tf.convert_to_tensor(np.array(data['rewards']),dtype=tf.float32)
        self.next_states=tf.convert_to_tensor(np.array(data['next_states']),dtype=tf.float32)
        self.dones=tf.convert_to_tensor(np.array(data['dones']),dtype=tf.float32)
        self.budgets=tf.convert_to_tensor(np.array(data['budgets']),dtype=tf.float32)
        self.aimcpcs=tf.convert_to_tensor(np.array(data['aimcpcs']),dtype=tf.float32)
        self.size=self.states.shape[0]


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=(batch_size,))

        return (
            tf.gather(self.states,indices=ind),
            tf.gather(self.actions,indices=ind),
            tf.gather(self.next_states,indices=ind),
            tf.gather(self.rewards,indices=ind),
            tf.gather(self.dones,indices=ind),
            tf.gather(self.budgets,indices=ind),
            tf.gather(self.aimcpcs,indices=ind)
        )

# class LData_Sampler(object):
#     def __init__(self, data, device):
#         self.states=torch.from_numpy(np.array(data['states'])).float()
#         self.rewards=torch.from_numpy(np.array(data['rewards'])).float()
#         self.actions=torch.from_numpy(np.array(data['actions'])).float()
#         self.next_states=torch.from_numpy(np.array(data['next_states'])).float()
#         self.dones=torch.from_numpy(np.array(data['dones'])).float()
#         self.budgets=torch.from_numpy(np.array(data['budgets'])).float()
#         self.size=self.states.shape[0]
#         self.device=device
#
#     def sample(self, batch_size):
#         ind = torch.randint(0, self.size, size=(batch_size,))
#         return (
#             self.states[ind].to(self.device),
#             self.actions[ind].to(self.device),
#             self.next_states[ind].to(self.device),
#             self.rewards[ind].to(self.device),
#             self.dones[ind].to(self.device),
#
#             self.budgets[ind].to(self.device)
#         )


def str_to_float(listz):
    listz = listz.strip('[]')
    listz = listz.split(',')
    a=[float(z) for z in listz]
    # print(a)
    # print(type(a))
    return a



def read_file(filename,typefile='low'):
    fist_tag=False
    if typefile=='hight':
        dataset = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': [], 'aimcpcs': [],
                   'budgets': []}
        with open(filename,'r',encoding='utf-8') as f:
            context=f.readline()
            while context:
                s1, s2, s3, s4 = [], [], [], []
                context=context.rstrip()
                context_data=context.split('@')
                for i in range(len(context_data)):
                    if i>1 and i<37:
                        if i!=4 or i!=6:
                            cont=str_to_float(context_data[i])
                            s1.append(cont[0])
                            s2.append(cont[1])
                            s3.append(cont[2])
                            s4.append(cont[3])
                s=[s1,s2,s3,s4]
                a=str_to_float(context_data[38])
                r=str_to_float(context_data[41])
                b=str_to_float(context_data[43])
                aimcpc=str_to_float(context_data[4])

                for i in range(len(a)):
                    state=np.array(s[i])
                    action=np.array(a[i])
                    reward=np.array(r[i])
                    budget=np.array(b[i])
                    a_cpc=np.array(aimcpc[i])
                    if i+1!=len(a):
                        next_state=np.array(s[i+1])
                        done=0.
                    else:
                        next_state=state
                        done=1.

                    dataset['states'].append(state)
                    dataset['rewards'].append([reward])
                    dataset['actions'].append([action])
                    dataset['dones'].append([done])
                    dataset['next_states'].append(next_state)
                    dataset['budgets'].append([budget])
                    dataset['aimcpcs'].append([a_cpc])
                context=f.readline()

        return dataset
    else:
        dataset={'states':[],'rewards':[],'actions':[],'next_states':[],'dones':[],'budgets':[]}
        with open(filename, 'r', encoding='utf-8') as f:
            context = f.readline()
            while context:
                context = context.rstrip()
                context_data = context.split('@')
                s1=str_to_float(context_data[12])+str_to_float(context_data[13])+str_to_float(context_data[14])+str_to_float(context_data[15])+str_to_float(context_data[16])+str_to_float(context_data[17])+str_to_float(context_data[18])
                s2=str_to_float(context_data[101])+str_to_float(context_data[102])+str_to_float(context_data[103])+str_to_float(context_data[104])+str_to_float(context_data[105])+str_to_float(context_data[106])+str_to_float(context_data[107])
                action = str_to_float(context_data[179])
                reward = str_to_float(context_data[173])
                t=sum(reward)
                budget=str_to_float(context_data[34])
                dataset['states'].append(np.array(s1))
                dataset['rewards'].append(np.array([t]))
                dataset['actions'].append(np.array(action))
                dataset['next_states'].append(np.array(s2))
                dataset['dones'].append(np.array([1.]))
                dataset['budgets'].append(np.array(budget))
                context=f.readline()
        return dataset


def get_dataset(batch,data_type='hight'):
    if data_type=='hight':
        dateset = read_file('../data/hight.txt', typefile='hight')
        data_simple=HData_Sampler(dateset)
        return data_simple.sample(batch)
    else:
        return None
    # else:
    #     dateset = read_file('../data/low.txt', typefile='low')
    #     data_simple=LData_Sampler(dateset)
    #     return data_simple.sample(batch)

if __name__ == '__main__':
    state,action,next_sate,reward,done,buget,cpc=get_dataset(64,data_type='hight')
    print("state：",state.shape)
    print("action:",action.shape)
    print("next_state:",next_sate.shape)






