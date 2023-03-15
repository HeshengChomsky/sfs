import tensorflow as tf
from sava_dataset import get_dataset,read_file,HData_Sampler
from HRA import Diffusion_QL

def train_hight_agent():
    dateset = read_file('../data/hight.txt', typefile='hight')
    data_simple = HData_Sampler(dateset)
    state, action, next_sate, reward, done, budget, aimcpc = data_simple.sample(64)
    state_dim=state.shape[1]
    action_dim=action.shape[1]
    agent=Diffusion_QL(state_dim=state_dim,action_dim=action_dim,max_action=1.,discount=0.98,tau=0.005)
    # actions=agent.sample_action(state)
    # agent.train(data_simple,10,batch_size=64)
    # print(actions.shape)
    loss_metric = agent.train(data_simple,iterations=2000,batch_size=64)
    print(loss_metric['actor_loss'])
    print(loss_metric['bc_loss'])
    print(loss_metric['critic_loss'])

if __name__ == '__main__':
    train_hight_agent()