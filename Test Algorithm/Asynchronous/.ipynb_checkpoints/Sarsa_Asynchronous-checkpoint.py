import gym
import random
import copy 
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import psutil
from torch.utils.tensorboard import SummaryWriter

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.99
n_train_processes = 16
update_interval = 5
max_train_ep = 10000
max_test_ep = 30000
target_update_interval = 100

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
            
def train(global_model, target_model, rank):

    env = gym.make('CartPole-v1')
    
    target_model.load_state_dict(global_model.state_dict())
    
    optimizer = optim.AdamW(global_model.parameters(), lr=learning_rate)
    
    for n_epi in range(max_train_ep):
        done = False
        s = env.reset()

        exploration = random.uniform(0.05, 0.15)
        epsilon = max(0.002, exploration - 0.01*(n_epi/3000)) #Linear annealing from 8% to 1%
        
        while not done:
            s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
            for t in range(update_interval):
            
                a = global_model.sample_action(torch.from_numpy(s).float(), epsilon) 
                #array, float, bool, _
                s_prime, r, done, info = env.step(a)
                done_mask = 0.0 if done else 1.0

                s_lst.append(s.tolist())
                a_lst.append([a])
                r_lst.append([r/100.0])
                s_prime_lst.append(s_prime.tolist())
                done_lst.append([done_mask])
          
                s = s_prime.copy()
    
                if done:
                    break

            a_lst = torch.tensor(a_lst, dtype=torch.float)
            r_lst = torch.tensor(r_lst, dtype=torch.float)
        
            td_target_lst = []
            #r + gamma*q(s',a') - q(s,a)
            _q_prime = target_model(torch.tensor(s_prime_lst, dtype=torch.float)) #global 이 아니라 fixed target network

            a_prime_lst = []
            for index in range(len(s_prime_lst)):
                _a_prime = global_model.sample_action(torch.tensor(s_prime_lst[index]).float(), epsilon) #action sampling: global vs target
                a_prime_lst.append([_a_prime])    
            
            a_prime_lst = torch.tensor(a_prime_lst, dtype=torch.float)
            q_prime_lst = _q_prime.gather(1,a_prime_lst.long())
            target = r_lst + gamma * q_prime_lst * done_mask

            q_value = global_model(torch.tensor(s_lst, dtype=torch.float)).gather(1,a_lst.long())
            
            loss = F.smooth_l1_loss(q_value, target.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if n_epi % target_update_interval==0 and rank == 1:
            print("NETWORK UPDATE...!")
            target_model.load_state_dict(global_model.state_dict())
            
    env.close()
    print("Training process {} reached maximum episode.".format(rank))

def test(global_model):
    #tensorboard
    writer = SummaryWriter('./runs/Sarsa/')
    
    env = gym.make('CartPole-v1')
    score = 0.0
    print_interval = 100

    for n_epi in range(max_test_ep):
        done = False
        s = env.reset()
        
        while not done:
            a = global_model(torch.from_numpy(s)).argmax().item()
            #array, float, bool, _
            s_prime, r, done, info = env.step(a)
            s = s_prime.copy()
            score += r

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(
                n_epi, score/print_interval))
            #tensorboard
            writer.add_scalar("Score", score/print_interval, n_epi)
            
            score = 0.0
            time.sleep(1)
    env.close()

if __name__ == '__main__':
    cpu_count = psutil.cpu_count()
    print(f"CPU 코어 수: {cpu_count}")
    
    mp.set_start_method('spawn')
    
    global_model = Qnet()
    global_model.share_memory()

    target_model = Qnet()
    target_model.share_memory()
    
    processes = []
    for rank in range(n_train_processes + 1):  # + 1 for test process
        if rank == 0:
            p = mp.Process(target=test, args=(global_model, ))
        else:
            p = mp.Process(target=train, args=(global_model, target_model, rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
