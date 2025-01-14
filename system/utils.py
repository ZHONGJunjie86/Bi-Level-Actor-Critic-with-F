import torch
import torch.nn as nn
import torch.multiprocessing as mp
from config.config import agent_type_list, obs_shape_by_type, args
import copy
import numpy as np
import os
import sys
# torch.set_default_tensor_type(torch.DoubleTensor)



class Shared_grad_buffers(nn.Module):
    def __init__(self, models, main_device, agent_type, agent_name):
        super(Shared_grad_buffers, self).__init__()
        self.device = main_device
        self.grads = {}
        self.agent_type = agent_type
        self.agent_name = agent_name
        for name, p in models.named_parameters():
            self.grads[name + '_grad'] = torch.zeros(p.size()).share_memory_().to(self.device)


    def add_gradient(self, models):
        for name, p in models.named_parameters():
            # print("name, p",name,p)
            # print("self.agent_type, self.agent_name",self.agent_type, self.agent_name)
            self.grads[name + '_grad'] += p.grad.data.to(self.device)
            # print("--------------------------------------------------------")

    def reset(self):
        for name, grad in self.grads.items():
            self.grads[name].fill_(0)


class Shared_Data(): #nn.Module
    def __init__(self, model_dict, load_path, save_path, main_device):
        # super(Shared_Data, self).__init__()
        self.shared_lock = mp.Manager().Lock()
        self.event = mp.Event()
        self.shared_count = mp.Value("d", 0)
        self.agent_name_list = ["leader", "follower"]
        self.device = main_device
        self.obs_shape = obs_shape_by_type
        self.hidden_size = 64
        self.action_dim = {"leader":5, "follower":1}
        self.clip = args.clip
        self.a_lr = args.a_lr
        self.episode = 1
        self.min_lr = args.min_lr
        self.lr_decay = args.lr_decay
        
        
        self.model_dict = model_dict
        self.load_path = load_path
        self.save_path = save_path

        self.shared_model = {}
        self.actor_optimizer =  {}
        for agent_type in agent_type_list:
            self.shared_model[agent_type] = {}
            self.actor_optimizer[agent_type] = {}
            for name in ["leader", "follower"]:
                self.model_dict[agent_type][name] = copy.deepcopy(self.model_dict[agent_type][name]).to(self.device).share_memory()
                self.shared_model[agent_type][name] = \
                    Shared_grad_buffers(model_dict[agent_type][name], 
                    main_device, agent_type, name).share_memory()
                self.actor_optimizer[agent_type][name] = \
                    torch.optim.Adam(self.model_dict[agent_type][name].parameters(),
                                                          lr=self.a_lr)
        

        # share training data
        self.share_training_data = {}
        for agent_type in agent_type_list:
            self.share_training_data[agent_type] = {}
            self.share_training_data[agent_type]["old_states"] = mp.Manager().list([])
            self.share_training_data[agent_type]["leader_action_behaviour"] = mp.Manager().list([])
            self.share_training_data[agent_type]["follower_share_info"] = mp.Manager().list([])
            for name in ["leader", "follower"]:
                self.share_training_data[agent_type][name] = {}
                self.share_training_data[agent_type][name]["old_hiddens"] = mp.Manager().list([])
                self.share_training_data[agent_type][name]["old_logprobs"] = mp.Manager().list([])
                self.share_training_data[agent_type][name]["advantages"] = mp.Manager().list([])
                self.share_training_data[agent_type][name]["target_value"] =  mp.Manager().list([])
                self.share_training_data[agent_type][name]["action"] = mp.Manager().list([])
        # self.share_training_data = mp.Manager().dict(self.share_training_data)
        # print("self.share_training_data",self.share_training_data)
        
        # loss_dict
        self.loss_name_list = ["a_loss", "c_loss", "entropy"]
        self.loss_dic = {}
        for agent_type in agent_type_list:
            self.loss_dic[agent_type] = {}
            for loss_name in self.loss_name_list:
                self.loss_dic[agent_type][loss_name] = {}
                for agent_name in self.agent_name_list:
                    self.loss_dic[agent_type][loss_name][agent_name] = mp.Value("d", 0.0)
    
    
    def reset(self):
        for agent_type in agent_type_list:
            for name in ["leader", "follower"]:
                self.shared_model[agent_type][name].reset()

    def save(self):
        for agent_type in agent_type_list:
            for name in ["leader", "follower"]:
                model_actor_path = self.save_path[agent_type] + agent_type  + name + ".pth"
                torch.save(self.model_dict[agent_type][name].state_dict(), model_actor_path)

    def load(self):
        for agent_type in agent_type_list:
            for name in self.agent_name_list:
                model_actor_path = self.load_path[agent_type]+ agent_type  + name + ".pth"
                #print(f'Actor path: {model_actor_path}')
                if  os.path.exists(model_actor_path):
                    actor = torch.load(model_actor_path, map_location=self.device)
                    self.model_dict[agent_type][name].load_state_dict(actor)
                    #print("Model loaded!")
                else:
                    sys.exit(f'Model not founded!')    
    
    def update_share_data(self, dict):
        for agent_type in agent_type_list:
            # if agent_type=="adversary":
            #     print("sum in center-----------",sum(dict[agent_type]["old_states"]))
            self.share_training_data[agent_type]["leader_action_behaviour"].extend(dict[agent_type]["leader_action_behaviour"])
            self.share_training_data[agent_type]["old_states"].extend(dict[agent_type]["old_states"])
            self.share_training_data[agent_type]["follower_share_info"].extend(dict[agent_type]["follower_share_info"])
            for name in ["leader", "follower"]:
                for list_name in self.share_training_data[agent_type][name].keys():
                    self.share_training_data[agent_type][name][list_name].extend(dict[agent_type][name][list_name])
        # print("adversary follower action ",len(self.share_training_data["adversary"]["follower"]["action"]))

    def get_shared_data(self):
        return_dict = {}
        for agent_type in agent_type_list:
            return_dict[agent_type] = {}
            return_dict[agent_type]["old_states"] = copy.deepcopy(list(self.share_training_data[agent_type]["old_states"]))
            return_dict[agent_type]["leader_action_behaviour"] = list(copy.deepcopy(self.share_training_data[agent_type]["leader_action_behaviour"]))
            return_dict[agent_type]["follower_share_info"] = list(copy.deepcopy(self.share_training_data[agent_type]["follower_share_info"]))
            for name in ["leader", "follower"]:
                return_dict[agent_type][name] = {}
                return_dict[agent_type][name]["old_hiddens"] = list(copy.deepcopy(self.share_training_data[agent_type][name]["old_hiddens"]))
                return_dict[agent_type][name]["old_logprobs"] = list(copy.deepcopy(self.share_training_data[agent_type][name]["old_logprobs"]))
                return_dict[agent_type][name]["advantages"] = list(copy.deepcopy(self.share_training_data[agent_type][name]["advantages"]))
                return_dict[agent_type][name]["target_value"] = list(copy.deepcopy(self.share_training_data[agent_type][name]["target_value"] ))
                return_dict[agent_type][name]["action"] = list(copy.deepcopy(self.share_training_data[agent_type][name]["action"]) )
        return return_dict

    
    def update_lr(self):
        if self.episode!= 0 and self.episode % 12 == 0 and self.a_lr>self.min_lr:
            self.a_lr = max(self.a_lr  * self.lr_decay, self.min_lr)
            for agent_type in agent_type_list:
                for name in ["leader", "follower"]:
                    self.actor_optimizer[agent_type][name] = \
                        torch.optim.Adam(self.model_dict[agent_type][name].parameters(),
                                                            lr=self.a_lr)
    
    
    def reset_share_data(self):
        for agent_type in agent_type_list:
            del self.share_training_data[agent_type]["old_states"][:]
            del self.share_training_data[agent_type]["leader_action_behaviour"][:]
            del self.share_training_data[agent_type]["follower_share_info"][:]
            for name in ["leader", "follower"]:
                for list_name in self.share_training_data[agent_type][name].keys():
                    del self.share_training_data[agent_type][name][list_name][:]

    
    def train(self):
        # reset loss
        self.loss_name_list = ["a_loss", "c_loss", "entropy"]
        for agent_type in agent_type_list:
            for loss_name in self.loss_name_list:
                for agent_name in self.agent_name_list:
                    self.loss_dic[agent_type][loss_name][agent_name].value = 0 
        
        
        for agent_type in ["adversary","agent"]: # agent_type_list
            self.old_logprobs = {}
            self.old_actions = {}
            self.old_values = {}
            self.old_action_values = {}
            self.old_hiddens = {}
            self.target_value = {}
            self.advantages = {}
            share_data_dict = self.share_training_data[agent_type]
            
            self.old_states = torch.tensor(np.array(share_data_dict["old_states"])).view(-1,1,self.obs_shape[agent_type]).to(self.device)
            self.follower_share_info = torch.tensor(np.array(share_data_dict["follower_share_info"])).view(-1,1,self.hidden_size).to(self.device)
            # torch.cat([self.old_states[:-1],
            #                         torch.tensor(share_data_dict["old_state"]).view(-1,1,self.obs_shape).to(self.device)
            #                         ], 0)
            self.leader_action_behaviour = torch.tensor(np.array(share_data_dict["leader_action_behaviour"])).view(-1,1,1).to(self.device)
            
            for name in self.agent_name_list:
                self.old_hiddens[name] = torch.tensor(np.array(share_data_dict[name]["old_hiddens"])).view(-1,1,self.hidden_size).to(self.device)
                # torch.cat([self.old_hiddens[name][:-1].reshape(-1,1,self.hidden_size),
                #                        torch.tensor(share_data_dict["old_hidden"]).view(-1,1,self.hidden_size).to(self.device)
                #                     ], 0)
                # print("old_h!!!!!!!!!!!!!")
                self.old_logprobs[name] = torch.tensor(np.array(share_data_dict[name]["old_logprobs"])).view(-1,1,1).to(self.device)
                # torch.cat([self.old_logprobs[name][:-1],
                #                           torch.tensor(share_data_dict["old_logprobs"]).view(-1,1,1).to(self.device)
                #                     ], 0)
                # print("old_log!!!!!!!!!!!!")
                self.advantages[name] = torch.tensor(np.array(share_data_dict[name]["advantages"])).view(-1,1,1).to(self.device)
                self.advantages[name] = (self.advantages[name]- self.advantages[name].mean()) / (self.advantages[name].std() + 1e-6) 
                # torch.cat([self.advantages[name].detach(),
                #                         torch.tensor(share_data_dict["advantages"]).view(-1,1,1).to(self.device)
                #                     ], 0)
                # print("old_adv!!!!!!!!!!!!")
                self.target_value[name] = torch.tensor(np.array(share_data_dict[name]["target_value"])).view(-1,1,1).to(self.device)
                #self.target_value[name] = (self.target_value[name]- self.target_value[name].mean()) / (self.target_value[name].std() + 1e-6) 
                # torch.cat([self.target_value[name],
                #                         torch.tensor(share_data_dict["target_value"]).view(-1,1,1).to(self.device)
                #                     ], 0)
                # print("share_data_dict[name]key------------------",share_data_dict[name].keys())#["action"]
                # print("old_target!!!!!!!!!!!!")
                self.old_actions[name] = torch.tensor(np.array(share_data_dict[name]["action"])).view(-1,1,self.action_dim[name]).to(self.device)
                # print("old_action!!!!!!!!!!!!")
                batch_size = self.old_hiddens[name].size()[0]
                # print("batch_size!!!!!!!!!!!!")
                # print("old_actions_size-----------------",self.old_actions[name].size())
                # print(self.old_states.size(),        
                    #   self.old_hiddens[name].size(),
                    #   self.old_actions[name].size(), #
                    #   self.old_logprobs[name].size(), 
                    #   self.advantages[name].size(), 
                    #   #self.old_values[name].size(),#
                    #   self.target_value[name].size(),
                    #   self.old_actions[name].size(),
                    #   self.leader_action_behaviour.size()
                    #   )
            print("batch_size------",batch_size, "------------lr",self.a_lr)#self.old_hiddens[name].size())
                #return
            for name in ["leader", "follower"]: #   self.agent_name_list
                index = [i for i in range(batch_size)]
                np.random.shuffle(index)
                index_start = 0
                batch_sample = batch_size // args.K_epochs
                # if self.episode > 2000 and agent_type == "agent":
                #     continue
                if name == "follower" and agent_type == "agent": # 
                    continue
                for _ in range(args.K_epochs): # batch_size#
                    indices = torch.tensor(index[index_start:index_start+batch_sample],requires_grad=False)# torch.randint(batch_size, size=(batch_sample,), requires_grad=False)
                    old_states = self.old_states[indices]
                    old_hidden = self.old_hiddens[name].view(-1,1,self.hidden_size)[indices].view(1, -1, self.hidden_size).view(1, -1, self.hidden_size)#
                    old_logprobs = self.old_logprobs[name][indices]
                    advantages = self.advantages[name][indices].detach()##
                    target_value = self.target_value[name][indices]
                    
                    
                    # print("old_actions_size-----------------",self.old_actions[name].size())
                    # print(old_states.size(),        
                    #   old_hidden.size(),
                    #   self.old_actions["leader"][indices].size(), #
                    #   old_logprobs.size(), 
                    #   advantages.size(), 
                    #   #self.old_values[name].size(),#
                    #   target_value.size(),
                    #   self.old_actions["follower"][indices].size(),
                    #   self.leader_action_behaviour[indices].size()
                    #   )
                    # print("start------inference")
                    # print("start------self.model_dict[agent_type][name]",self.model_dict[agent_type][name])#self.old_hiddens[name].size())
                    logprobs, action_value, _, _, entropy = self.model_dict[agent_type][name](obs = old_states, h_old = old_hidden, 
                                                                    leader_action = self.old_actions["leader"][indices], #[indices]
                                                                    follower_action = self.old_actions["follower"][indices], #[indices]
                                                                    leader_behaviour = self.leader_action_behaviour[indices], 
                                                                    share_inform = self.follower_share_info[indices], train = True) #[indices]
            
                    ratios = torch.exp(logprobs.view(batch_sample,1,-1) - old_logprobs.detach())

                    surr1 = ratios*advantages
                    surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip)*advantages 
                    #Dual_Clip
                    surr3 = torch.max(torch.min(surr1, surr2),3*advantages)
                    #torch.min(surr1, surr2)#
                    
                    
                    # value_pred_clip = old_value.detach() +\
                    #     torch.clamp(action_value - old_value.detach(), -self.vf_clip_param, self.vf_clip_param)
                    # critic_loss1 = (action_value - target_value.detach()).pow(2)
                    # critic_loss2 = (value_pred_clip - target_value.detach()).pow(2)
                    # critic_loss = 0.5 * torch.max(critic_loss1 , critic_loss2).mean()
                    critic_loss = torch.nn.SmoothL1Loss()(action_value, target_value) #(action_value - target_value).pow(2).mean()#

                    actor_loss = -surr3.mean() - 0.015 * entropy + critic_loss #0.5 * 
                    # print("start------actor_loss",actor_loss)
                    
                    # # do the back-propagation...
                    self.model_dict[agent_type][name].zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer[agent_type][name].step()

                    self.loss_dic[agent_type]['a_loss'][name].value += float(surr3.mean().cpu().detach().numpy())
                    self.loss_dic[agent_type]['c_loss'][name].value += float(critic_loss.cpu().detach().numpy())
                    self.loss_dic[agent_type]['entropy'][name].value += float(entropy.cpu().detach().numpy())
                    index_start += batch_sample
        
        self.update_lr()
        self.episode += 1
        return self.loss_dic
        
        
    def get_loss_dict(self):
        return self.loss_dic