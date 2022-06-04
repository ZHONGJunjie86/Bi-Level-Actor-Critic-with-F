import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
import torch.nn as nn
from torch.autograd import Variable
from network import Actor
from utils import Memory

torch.set_default_tensor_type(torch.DoubleTensor)


class PPO:
    def __init__(self,  args, obs_shape, device):
        self.device = device
        self.a_lr = args.a_lr
        self.gamma = args.gamma

        # Initialise actor network 
        self.agent_name_list = ["leader", "follower"]
        self.action_dim = {"leader":5, "follower":1}
        self.actor = {name: Actor(self.action_dim[name], name).to(self.device) for name in self.agent_name_list}
        self.actor_optimizer = {name: torch.optim.Adam(self.actor[name].parameters(),
                                                          lr=self.a_lr) for name in self.agent_name_list}
        self.SmoothL1Loss = torch.nn.SmoothL1Loss()

        #
        self.memory ={name: Memory() for name in self.agent_name_list}
        self.hidden_state_zero = torch.zeros(1,1,64).to(self.device)
        self.loss_name_list = ["actor_loss_", "critic_loss_"]
        self.loss_dic = {loss_name + agent_name: 0 for agent_name in self.agent_name_list for loss_name in self.loss_name_list}
        
        #
        self.obs_shape = obs_shape
        self.eps_clip = 0.2
        self.vf_clip_param = 0.2
        self.lam = 0.95
        self.K_epochs = args.K_epochs
        self.old_value_1, self.old_value_2 = 0,0
        self.entropy_coef = 0.01

        # for K updates
        self.advantages = []
        self.target_value = []

    def bilevel_compare_action(self, obs_tensor, reward):
        leader_action_values = []
        for leader_action_index in range(self.action_dim["leader"]):
            leader_action = np.zeros(self.action_dim["leader"])
            leader_action[leader_action_index] = 1
            follower_logprob, follower_action_value, follower_action, follower_hidden_state = self.actor["follower"](obs_tensor, torch.tensor(leader_action))
            leader_logprob, leader_action_value, leader_hidden_state = \
                                    self.actor["leader"](obs_tensor, torch.tensor(leader_action, follower_action))
            leader_action_values.append([leader_action_value, follower_action_value,
                                         leader_action_index, follower_action,
                                         leader_logprob, follower_logprob,
                                         leader_hidden_state, follower_hidden_state])

        # sort
        leader_action_values.sort(key = lambda x: x[0], reversed = True)
        # return
        action = {}
        action_logprob = {}
        h_s = {}
        action_value = {}
        value = {}
        action_value["leader"] = leader_action_values[0][0]
        action_value["follower"] = leader_action_values[0][1]
        action["leader"] = leader_action_values[0][2]
        action["follower"] = leader_action_values[0][3]
        action_logprob["leader"] = leader_action_values[0][4]
        action_logprob["follower"] = leader_action_values[0][6]
        h_s["leader"] = leader_action_values[0][7]
        h_s["follower"] = leader_action_values[0][8]
        value["leader"] = 0
        value["follower"] = 0
        for i in leader_action_values:
            value["leader"] += leader_action_values[0][0]
            value["follower"] += leader_action_values[0][1]
        value["leader"] = value["leader"]/self.action_dim["leader"]
        value["follower"] = value["follower"]/self.action_dim["leader"]

        reward_dic = {"leader": reward, "follower": self.compute_follower_reward(reward)}

        return action_value, action, action_logprob, h_s, value, reward_dic

    def compute_follower_reward(self, reward):
        pass

    def choose_action(self, state, reward, done):
        
        if len(self.memory["leader"].states) ==0:
            for name in self.agent_name_list:
                with torch.no_grad():
                    self.memory[name].hidden_state.append(self.hidden_state_zero.cpu().numpy())
        
        obs_tensor = torch.Tensor(state).to(self.device).reshape(1,self.obs_shape)        
        action_value, action, action_logprob, h_s, value, reward_dic = self.bilevel_compare_action(obs_tensor, reward)

        with torch.no_grad():
            self.memory["leader"].states.append(state)
            for name in self.agent_name_list:
                self.memory[name].actions.append(action[name].cpu().numpy())
                self.memory[name].logprobs.append(action_logprob[name].cpu().numpy()) 
                self.memory[name].hidden_state.append(h_s[name].cpu().numpy())
                self.memory[name].action_value.append(action_value[name].cpu().numpy())
                self.memory[name].value.append(value[name].cpu().numpy())
                self.memory[name].rewards.append(reward_dic[name])
        
        return action["leader"].reshape(1,1).cpu().detach().numpy()[0]


    def compute_GAE(self, training_time, main_process = False):
        
        if training_time ==0:
            batch_size_1 = torch.tensor(self.memory_our_enemy[0].logprobs).view(-1, 2, 1).size()[0]
            batch_size_2 = torch.tensor(self.memory_our_enemy[1].logprobs).view(-1, 2, 1).size()[0]
            self.old_states = torch.cat( 
                            [torch.tensor(self.memory_our_enemy[0].states).view(-1,4,30,30),
                            torch.tensor(self.memory_our_enemy[1].states).view(-1,4,30,30)
                            ], 0).to(self.device).detach() 
            self.old_logprobs = torch.cat(
                            [torch.tensor(self.memory_our_enemy[0].logprobs).view(-1, 2, 1),
                            torch.tensor(self.memory_our_enemy[1].logprobs).view(-1, 2, 1)
                            ], 0).to(self.device).detach()
            self.old_actions = torch.cat( 
                            [torch.tensor(self.memory_our_enemy[0].actions).view(-1, 2, 1),
                            torch.tensor(self.memory_our_enemy[1].actions).view(-1, 2, 1)
                            ], 0).to(self.device).detach() 

            self.old_num_vector = torch.cat(
                            [torch.tensor(self.memory_our_enemy[0].num_vectors).view( -1,1, self.num_vector_length),
                            torch.tensor(self.memory_our_enemy[1].num_vectors).view(-1,1, self.num_vector_length)
                            ], 0).to(self.device).detach() 
            self.old_value = torch.cat(
                            [torch.tensor(self.memory_our_enemy[0].value).view(-1, 1, 1),
                            torch.tensor(self.memory_our_enemy[1].value).view(-1, 1, 1)
                            ], 0).to(self.device).detach()    
            self.old_hidden = torch.cat( 
                            [torch.tensor(self.memory_our_enemy[0].hidden_state[:-1]).view(-1, 1, 64),
                            torch.tensor(self.memory_our_enemy[1].hidden_state[:-1]).view(-1, 1, 64)
                            ], 0).to(self.device).detach() 
            compute_rewards = torch.cat( 
                            [torch.tensor(self.memory_our_enemy[0].rewards).view(-1, 1, 1),
                            torch.tensor(self.memory_our_enemy[1].rewards).view(-1, 1, 1)
                            ], 0).to(self.device).detach() 
            compute_termi = torch.cat( 
                            [torch.tensor(self.memory_our_enemy[0].is_terminals).view(-1, 1, 1)[-batch_size_1:],
                            torch.tensor(self.memory_our_enemy[1].is_terminals).view(-1, 1, 1)[-batch_size_2:]
                            ], 0).to(self.device).detach()
            # Monte Carlo estimate of rewards:
            rewards = []
            GAE_advantage = []
            target_value = []
            #
            discounted_reward = 0
            values_pre = 0
            advatage = 0

            for reward, is_terminal,values in zip(reversed(compute_rewards), reversed(compute_termi),
                                        reversed(self.old_value)): #反转迭代
                
                values = values.cpu().detach().numpy()
                reward = reward.cpu().detach().numpy()
                is_terminal = is_terminal.cpu().detach().numpy()

                discounted_reward = reward +  self.gamma *discounted_reward 
                rewards.insert(0, discounted_reward) #插入列表

                
                delta = reward + (1-is_terminal)*self.gamma*values_pre - values  
                advatage = delta + self.gamma*self.lam*advatage * (1-is_terminal)
                GAE_advantage.insert(0, advatage) #插入列表
                target_value.insert(0,float(values + advatage))
                values_pre = values
            
            # Normalizing the rewards:
            rewards = torch.tensor(rewards).to(self.device).view(-1,1,1)
            self.target_value = torch.tensor(target_value).to(self.device).view(-1,1,1)
            GAE_advantage = torch.tensor(GAE_advantage).to(self.device).view(-1,1,1)
            self.advantages = (GAE_advantage- GAE_advantage.mean()) / (GAE_advantage.std() + 1e-6) 
            
        #compute
        batch_size = self.target_value.size()[0]
        batch_sample = batch_size#int(batch_size / self.K_epochs)
        indices = torch.randint(batch_size, size=(batch_sample,), requires_grad=False)#, device=self.device

        old_states = self.old_states[indices]
        old_num_vector = self.old_num_vector.reshape(batch_size,1,self.num_vector_length)[indices].view(1,-1, self.num_vector_length)
        old_hidden = self.old_hidden.reshape(batch_size,1,64)[indices].view(1,-1, 64)
        old_actions = self.old_actions[indices]
        old_logprobs = self.old_logprobs[indices]
        advantages = self.advantages[indices].detach()
        old_value = self.old_value[indices]
        target_value = self.target_value[indices]


        _, logprobs, dist_entropy, _, value = self.actor(old_states, old_num_vector, old_hidden, old_actions)
 
        ratios = torch.exp(logprobs.view(-1,2,1).sum(1,keepdim = True) - 
                           old_logprobs.view(-1,2,1).sum(1,keepdim = True).detach())

        surr1 = ratios*advantages
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)*advantages 
        #Dual_Clip
        surr3 = torch.max(torch.min(surr1, surr2),3*advantages)
        #torch.min(surr1, surr2)#
        
        value_pred_clip = old_value.detach() +\
            torch.clamp(value -old_value.detach(), -self.vf_clip_param, self.vf_clip_param)#self.vf_clip_param
        critic_loss1 = (value - target_value.detach()).pow(2)
        critic_loss2 = (value_pred_clip - target_value.detach()).pow(2)
        critic_loss = 0.5 * torch.max(critic_loss1 , critic_loss2).mean()
        #critic_loss = torch.nn.SmoothL1Loss()(state_values_1, target_value) + torch.nn.SmoothL1Loss()(state_values_2, target_value)

        actor_loss = -surr3.mean() - self.entropy_coef * dist_entropy + 0.5 * critic_loss
        
        # do the back-propagation...
        self.actor.zero_grad()
        actor_loss.backward()

        if main_process:
            self.loss_dic = [actor_loss, critic_loss]
        else:
            self.hidden_state = torch.zeros(1,1,64).to(self.device)

        self.a_loss += float(actor_loss.cpu().detach().numpy())
        self.c_loss += float(critic_loss.cpu().detach().numpy())
        return actor_loss.detach(), critic_loss.detach()

    def add_gradient(self, shared_grad_buffer):
        # add the gradient to the shared_buffer...
        shared_grad_buffer.add_gradient(self.actor)


    def update(self, shared_grad_buffer_grads, worker_num):     

        self.actor.zero_grad()
        for n, p in self.actor.named_parameters():
            p.grad = Variable(shared_grad_buffer_grads[n + '_grad'])
        
        nn.utils.clip_grad_norm_(self.actor.parameters(),5)
        self.actor_optimizer.step()

        self.hidden_state = torch.zeros(1,1,64).to(self.device)

        return self.c_loss, self.a_loss

    def get_loss(self):
        return self.c_loss, self.a_loss

    def get_actor(self):
        return self.actor

    def reset_loss(self):
        self.a_loss = 0
        self.c_loss = 0

    def copy_memory(self, sample_mem):
        self.memory_our_enemy = sample_mem
    
    def clear_memory(self):
        self.memory_our_enemy[0].clear_memory()
        self.memory_our_enemy[1].clear_memory()

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        print("base_path",base_path)

        model_actor_path = os.path.join(base_path, "actor_"  + ".pth")
        print(f'Actor path: {model_actor_path}')

        if  os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=self.device)
            self.actor.load_state_dict(actor)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def save_model(self, run_dir, episode):
        print("---------------save-------------------")
        base_path = os.path.join(run_dir, 'trained_model')
        print("base_path", base_path)
        print("new_lr: ",self.a_lr)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_"  + ".pth") #+ str(episode)
        torch.save(self.actor.state_dict(), model_actor_path)



