import os
from time import sleep
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
import torch.nn as nn
from torch.autograd import Variable
from algorithm.network import ActorCritic
from algorithm.utils import Memory
import torch.nn.functional as F
import copy
torch.set_default_tensor_type(torch.DoubleTensor)


class PPO:
    def __init__(self,  args, obs_shape, device, agent_type):
        self.device = device
        self.a_lr = args.a_lr
        self.gamma = args.gamma
        self.agent_type = agent_type

        #
        self.obs_shape = obs_shape
        self.eps_clip = 0.2
        self.vf_clip_param = 0.2
        self.lam = 0.95
        self.K_epochs = args.K_epochs
        self.old_value_1, self.old_value_2 = 0,0
        self.entropy_coef = 0.01
        self.hidden_size = 64
        self.hidden_state_zero = torch.zeros(1,1,self.hidden_size).to(self.device)

        # social reward coef
        self.env_coef = 0.5
        self.social_coef = 0.5
        self.reward_follower_last = 0

        # for K updates
        self.advantages = {}
        self.target_value = {}
        
        # Initialise actor network 
        self.agent_name_list = ["leader", "follower"]
        self.action_dim = {"leader":5, "follower":1}

        
        self.actor = {name: copy.deepcopy(ActorCritic(self.obs_shape, self.action_dim["leader"], 
                                       self.action_dim["follower"], 
                                       name)).to(self.device) 
                                       for name in self.agent_name_list}

        
        self.actor_optimizer = {name: copy.deepcopy(torch.optim.Adam(self.actor[name].parameters(),
                                                          lr=self.a_lr)) for name in self.agent_name_list}
        self.SmoothL1Loss = torch.nn.SmoothL1Loss()

        #
        self.memory ={name: Memory() for name in self.agent_name_list}
        self.loss_name_list = ["a_loss", "c_loss"]
        self.loss_dic = {}
        for loss_name in self.loss_name_list:
            self.loss_dic[loss_name] = {}
            for agent_name in self.agent_name_list:
                self.loss_dic[loss_name][agent_name] = 0 

        #
        self.reset()


    def bilevel_compare_action(self, obs_tensor, reward):
        data_dict_list = []
        leader_hidden = self.memory["leader"].hidden_states[-1]
        follower_hidden = self.memory["follower"].hidden_states[-1]
        for leader_action_index in range(self.action_dim["leader"]):
            leader_action = np.zeros(self.action_dim["leader"])
            leader_action[leader_action_index] = 1

            with torch.no_grad():
                follower_logprob, follower_action_value, follower_action, follower_hidden_state, _ =\
                                        self.actor["follower"](obs_tensor, torch.tensor(follower_hidden).to(self.device), 
                                                            torch.tensor(leader_action).to(self.device))
                
                leader_logprob, leader_action_value, leader_action_behaviour, leader_hidden_state, _ = \
                                        self.actor["leader"](obs_tensor, torch.tensor(leader_hidden).to(self.device), 
                                                            torch.tensor(leader_action).detach().to(self.device), 
                                                            torch.tensor(follower_action).detach().to(self.device))
            
            leader_action_behaviour_numpy = np.zeros(self.action_dim["leader"])
            leader_action_behaviour_numpy[leader_action_behaviour] = 1
            
            data_dict_list.append({"action_value":{"leader":leader_action_value, "follower": follower_action_value},
                                    "action":{"leader":leader_action_behaviour_numpy, "follower": follower_action,
                                              "leader_index":
                                                            #  leader_action_index
                                                             leader_action_behaviour
                                                             },
                                    "action_logprob":{"leader":leader_logprob, "follower": follower_logprob},
                                     "h_s":{"leader": leader_hidden_state, "follower": follower_hidden_state}
                                })

        # return
        value_dic = {"leader": 0, "follower": 0}
        for i in data_dict_list:
            value_dic["leader"] += i["action_value"]["leader"]
            value_dic["follower"] += i["action_value"]["follower"]
        value_dic["leader"] = value_dic["leader"]/self.action_dim["leader"]
        value_dic["follower"] = value_dic["follower"]/self.action_dim["leader"]

        # sort
        data_dict_list.sort(key = lambda x: x["action_value"]["leader"], reverse = True)
        return_dict = data_dict_list[0]
        return_dict["value"] = {"leader": value_dic["leader"], "follower": value_dic["follower"]}

        # reward_shaping
        return_dict["reward"] = {"leader": reward, 
                                "follower": self.compute_follower_reward(reward, 
                                                                        return_dict["action_value"]["leader"],
                                                                        return_dict["value"]["leader"]
                                                                        )}
        return return_dict


    def choose_action(self, state, reward, done):
        
        obs_tensor = torch.Tensor(state).to(self.device).reshape(1,self.obs_shape)        
        data_dict = self.bilevel_compare_action(obs_tensor, reward)

        with torch.no_grad():
            self.memory["leader"].states.append(state)
            self.memory["leader"].is_terminals.append(done)
            self.memory["leader"].leader_action_behaviour.append(data_dict["action"]["leader_index"])
            for name in self.agent_name_list:
                self.memory[name].actions.append(data_dict["action"][name])
                self.memory[name].logprobs.append(data_dict["action_logprob"][name].cpu().numpy()) 
                self.memory[name].hidden_states.append(data_dict["h_s"][name].cpu().numpy())
                self.memory[name].action_values.append(data_dict["action_value"][name].cpu().numpy())
                self.memory[name].values.append(data_dict["value"][name].cpu().numpy())
                self.memory[name].rewards.append(data_dict["reward"][name])
        
        return int(data_dict["action"]["leader_index"])

    
    def compute_follower_reward(self, reward, leader_action_value, leader_state_value):
        leader_adv =  leader_action_value - leader_state_value
        reward_follower = self.social_coef * self.reward_follower_last + self.entropy_coef * reward 
        self.reward_follower_last = leader_adv
        return reward_follower


    def compute_loss(self, training_time, main_process = False):
        if training_time ==0:
            with torch.no_grad():
                self.old_states = torch.tensor(np.array(self.memory["leader"].states)
                                            ).view(-1,1,self.obs_shape).to(self.device)
                self.old_compute_termi = torch.tensor(self.memory["leader"].is_terminals).to(self.device).detach() 
                self.leader_action_behaviour = torch.tensor(self.memory["leader"].leader_action_behaviour).to(self.device).detach() 
                self.old_logprobs = {}
                self.old_actions = {}
                self.old_values = {}
                self.old_action_values = {}
                self.old_hiddens = {}
                self.compute_rewards= {}
                for name in self.agent_name_list:
                    self.old_logprobs[name] = torch.tensor(self.memory[name].logprobs
                                                            ).view(-1, 1, 1).to(self.device)
                    self.old_actions[name] = torch.tensor(self.memory[name].actions
                                                         ).view(-1, 1, self.action_dim[name]).to(self.device)
                    self.old_values[name] = torch.tensor(self.memory[name].values).view(-1, 1, 1).to(self.device)
                    self.old_action_values[name] = torch.tensor(self.memory[name].action_values).view(-1, 1, 1).to(self.device)
                    self.old_hiddens[name] = torch.tensor(self.memory[name].hidden_states[:-1]).to(self.device).detach() 
                    self.compute_rewards[name] = torch.tensor(self.memory[name].rewards[1:]).to(self.device).detach() 
                    
        
        for name in self.agent_name_list:
            self.compute_GAE(self.compute_rewards[name], self.old_compute_termi, training_time, name)
            #compute
            batch_size = self.old_hiddens[name].size()[0]
            batch_sample = batch_size#int(batch_size / self.K_epochs) # 
            indices = torch.randint(batch_size, size=(batch_sample,), requires_grad=False)#, device=self.device

            # print(self.old_states.size(),
            #       self.old_hiddens[name].size(),
            #       self.old_actions[name].size(),
            #       self.old_logprobs[name].size(), #
            #       self.advantages[name].size(), #
            #       self.old_values[name].size(),
            #       self.target_value[name].size() 
            #       )

            old_states = self.old_states[indices]
            old_hidden = self.old_hiddens[name].reshape(-1,1,self.hidden_size)[indices].view(1, -1, self.hidden_size)
            old_logprobs = self.old_logprobs[name][indices]
            advantages = self.advantages[name][indices].detach()
            old_value = self.old_values[name][indices]
            target_value = self.target_value[name][indices]


            logprobs, action_value, _, _, entropy = self.actor[name](old_states, old_hidden, 
                                                            self.old_actions["leader"][indices], 
                                                            self.old_actions["follower"][indices],
                                                            self.leader_action_behaviour[indices], train = True)
    
            ratios = torch.exp(logprobs.view(batch_sample,1,-1) - old_logprobs.detach())

            surr1 = ratios*advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)*advantages 
            #Dual_Clip
            surr3 = torch.max(torch.min(surr1, surr2),3*advantages)
            #torch.min(surr1, surr2)#
            
            
            value_pred_clip = old_value.detach() +\
                torch.clamp(action_value - old_value.detach(), -self.vf_clip_param, self.vf_clip_param)
            critic_loss1 = (action_value - target_value.detach()).pow(2)
            critic_loss2 = (value_pred_clip - target_value.detach()).pow(2)
            critic_loss = 0.5 * torch.max(critic_loss1 , critic_loss2).mean()
            # critic_loss = torch.nn.SmoothL1Loss()(action_value, target_value) 
            
            
            # if name == "leader":
            #     actor_loss = -surr3.mean()  - self.entropy_coef * entropy  + 0.5 * critic_loss
            # else:
            #     actor_loss = -surr3.mean() - self.entropy_coef * entropy + 0.5 * critic_loss
            actor_loss = -surr3.mean() - self.entropy_coef * entropy + 0.5 * critic_loss
            
            
            # do the back-propagation...
            self.actor[name].zero_grad()
            actor_loss.backward()

            self.loss_dic['a_loss'][name] += float(actor_loss.cpu().detach().numpy())
            self.loss_dic['c_loss'][name] += float(critic_loss.cpu().detach().numpy())
            
        # return 0, 0 !!!!!!!!!!!!! return too earily
        

    def compute_GAE(self, compute_rewards, compute_termi, training_time, name):
        if training_time ==0:
            # Monte Carlo estimate of rewards:
            rewards = []
            GAE_advantage = [self.memory[name].action_values[-1]]
            target_value = [self.memory[name].values[-1]]  # 补一个最后的?
            #
            discounted_reward = 0
            action_value_pre = None
            advatage = 0

            for reward, is_terminal, value, action_value in zip(reversed(compute_rewards), reversed(compute_termi),
                                                  reversed(self.memory[name].values), reversed(self.memory[name].action_values)): #反转迭代
                
                reward = reward.cpu().detach().numpy()
                is_terminal = is_terminal.cpu().detach().numpy()

                discounted_reward = reward +  self.gamma *discounted_reward
                rewards.insert(0, discounted_reward) #插入列表

                if action_value_pre == None:   #  最后补一个
                    action_value_pre = action_value
                delta = reward + (1-is_terminal)*self.gamma*action_value_pre - value  
                advatage = delta + self.gamma*self.lam*advatage * (1-is_terminal)
                GAE_advantage.insert(0, advatage) #插入列表
                target_value.insert(0,float(value + advatage))
                action_value_pre = action_value
            
            # Normalizing the rewards:
            rewards = torch.tensor(rewards).to(self.device).view(-1,1,1)
            self.target_value[name] = torch.tensor(target_value).to(self.device).view(-1,1,1)
            GAE_advantage = torch.tensor(GAE_advantage).to(self.device).view(-1,1,1)
            self.advantages[name] = (GAE_advantage- GAE_advantage.mean()) / (GAE_advantage.std() + 1e-6) 
        
        return None
            

    def add_gradient(self, shared_model_dict):
        # add the gradient to the shared_buffer...
        for name in self.agent_name_list:
            shared_model_dict[self.agent_type][name].add_gradient(self.actor[name])


    def update(self, grads_dict):     
        for name in self.agent_name_list:
            self.actor[name].zero_grad()
            
            for n, p in self.actor[name].named_parameters():
                p.grad = Variable(grads_dict[self.agent_type][name].grads[n + '_grad'])
                
            nn.utils.clip_grad_norm_(self.actor[name].parameters(),5)
            self.actor_optimizer[name].step()


    def get_actor(self):
        return self.actor

    
    def reset(self):
        
        for name in self.agent_name_list:
            self.memory[name].clear_memory()
            
        if len(self.memory["leader"].states) ==0:
            for name in self.agent_name_list:
                with torch.no_grad():
                    self.memory[name].hidden_states.append(self.hidden_state_zero.cpu().numpy())
        
        for loss_name in self.loss_name_list:
            self.loss_dic[loss_name] = {}
            for agent_name in self.agent_name_list:
                self.loss_dic[loss_name][agent_name] = 0 

    def load_model(self, model_load_path):

        # "path + agent/adversary + leader/follower + .pth"
        for name in self.agent_name_list:
            model_actor_path = model_load_path[self.agent_type]+ self.agent_type  + name + ".pth"
            #print(f'Actor path: {model_actor_path}')
            if  os.path.exists(model_actor_path):
                actor = torch.load(model_actor_path, map_location=self.device)
                self.actor[name].load_state_dict(actor)
                #print("Model loaded!")
            else:
                sys.exit(f'Model not founded!')

    def save_model(self, model_save_path):
        # print("---------------save-------------------")
        # print("new_lr: ",self.a_lr)

        # "path + agent/adversary + leader/follower + .pth"
        for name in self.agent_name_list:
            model_actor_path = model_save_path[self.agent_type]+ self.agent_type  + name + ".pth"
            torch.save(self.actor[name].state_dict(), model_actor_path)

    def quick_load_model(self, new_model_dict):
        for name in self.agent_name_list:
            self.actor[name].load_state_dict(
                                             new_model_dict[self.agent_type][name].state_dict())

