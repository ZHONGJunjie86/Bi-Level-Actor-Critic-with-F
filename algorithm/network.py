from unicodedata import name
from numpy import tri
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
# torch.set_default_tensor_type(torch.DoubleTensor)


class ActorCritic(nn.Module):        
    def __init__(self, obs_size, leader_action_dim, follower_action_dim, name):  #(n+2p-f)/s + 1 
        super(ActorCritic, self).__init__()

        # share info
        self.self_attention = nn.MultiheadAttention(64, 4)        
        self.gru = nn.GRU(input_size = 64, hidden_size = 64, num_layers = 1, batch_first=True) 
        #self.gru.flatten_parameters()
        self.name = name
        self.leader_action_dim = leader_action_dim
        self.follower_action_dim = follower_action_dim
        self.obs_size = obs_size
        self.softmax = nn.Softmax(dim=-1)

        # actor
        if name == "follower":
            # state 包含 leader 的动作
            self.linear1 = nn.Linear(obs_size + leader_action_dim, 64)
            # actor beta distribution
            self.linear_alpha = nn.Linear(64, follower_action_dim)
            self.linear_beta = nn.Linear(64, follower_action_dim)
            self.beta_dis =  torch.distributions.beta.Beta
            self.normal_dis = torch.distributions.Normal
            # critic 含两个动作，follower是后算出来的
            self.linear_critic_1 = nn.Linear(64 + self.follower_action_dim * 2, 1)
        elif name == "leader":
            # state 
            self.linear1 = nn.Linear(obs_size, 64)
            # actor Categorical
            # 包含 follower的动作，自己的 one-hot（这俩决定SE） 
            self.linear_actor_combine = nn.Linear(64 + follower_action_dim + leader_action_dim, leader_action_dim)
            # self.linear_leader_logits = nn.Linear(64, leader_action_dim)
            self.categorical_dis = torch.distributions.Categorical
            # critic 含两个动作
            self.linear_critic_1 = nn.Linear(64 + follower_action_dim + leader_action_dim,  1)

        # critic == Q
        #self.linear_critic_2 = nn.Linear(64, 1)
 
        self.initialize()

    
    def initialize(self):
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        # torch.nn.init.zeros_(self.linear1.bias)
        # torch.nn.init.kaiming_uniform_(self.self_attention.weight)
        # torch.nn.init.kaiming_uniform_(self.gru.weight)

        if self.name == "leader":
            torch.nn.init.kaiming_normal_(self.linear_actor_combine.weight)
            # torch.nn.init.zeros_(self.linear_leader_logits.bias)
        elif self.name == "follower":
            torch.nn.init.kaiming_normal_(self.linear_alpha.weight)
            # torch.nn.init.zeros_(self.linear_alpha.bias)
            torch.nn.init.kaiming_normal_(self.linear_beta.weight)
            # torch.nn.init.zeros_(self.linear_beta.bias)

        torch.nn.init.kaiming_normal_(self.linear_critic_1.weight)
        # torch.nn.init.zeros_(self.linear_critic_1.bias)
        #torch.nn.init.kaiming_normal_(self.linear_critic_2.weight)
        # torch.nn.init.zeros_(self.linear_critic_2.bias)
        

    def forward(self, obs, h_old = None,  
                leader_action = None, follower_action = False, 
                leader_behaviour = None, train = False): 
        # share info
        batch_size = obs.size()[0]
        
        # if train:
        #     print("obs.size(),leader_action.size(),follower_action.size()--------",obs.size(),leader_action.size(),follower_action.size())
        if self.name == "follower":
            obs = torch.cat([obs.view(batch_size, 1, -1), 
                                         leader_action.reshape(batch_size, 1, self.leader_action_dim)  # + 0.001
                            ], -1).view(batch_size, -1)
        elif self.name == "leader":
            obs = torch.cat([obs], -1).view(batch_size, 1, -1)
            
        # if train:
        #     print("after cat---------")
        x = F.relu(self.linear1(obs.float()))
        # print("self.linear1---", x)
        x = x.view(batch_size, 1, -1)
        # if train:
        #     print("after fc---------")
        x = self.self_attention(x,x,x)[0] + x
        # print("self.self_attention", x)
        # if train:
        #     print("after attebtion---------")
        x,h_state = self.gru(x, h_old.detach())
        # print("self.gru---",x)
        # if train:
        #     print("after gru---------")
        # actor
        if self.name == "follower":
            # print("follower x----------", x)
            mu = torch.tanh(self.linear_alpha(x))   
            sigma = torch.sigmoid(self.linear_beta(x)) + 1e-5  # >0
            dis =  self.normal_dis(mu.reshape(batch_size,1), sigma.reshape(batch_size,1))
            if train:
                action = follower_action.view(batch_size,1)
            else:
                action = dis.sample().clip(-1,1)
            entropy = dis.entropy().mean()
            selected_log_prob = dis.log_prob(action)
            
        elif self.name == "leader":
            combined_vector =  torch.cat([x.view(batch_size, 1, -1), leader_action.reshape(batch_size, 1, self.leader_action_dim).float(), 
                                  follower_action.reshape(batch_size, 1, self.follower_action_dim).float()], -1
                                  )
            combined_logits = self.softmax(self.linear_actor_combine(combined_vector))
            #logits = self.softmax(self.linear_leader_logits(combined_logits))
            dis =  self.categorical_dis(combined_logits.reshape(batch_size, 1, self.leader_action_dim))
            
            # if train:
            #     print("after categorical---------")
            if train:
                action = leader_behaviour.view(batch_size,-1)
            else:
                action = dis.sample()
            entropy = dis.entropy().mean()
            selected_log_prob = dis.log_prob(action)

        # critic == Q
        if self.name == "follower":
            # add_follower_act_logits = torch.cat([
            #                                      x.view(batch_size, -1), 
            #                                      mu.view(batch_size,-1), 
            #                                      sigma.view(batch_size,-1)
            #                                      ], -1).view(batch_size, -1)
            action_value = self.linear_critic_1(x.view(batch_size, -1))#F.relu()
            # action_value = self.linear_critic_2(x)
        elif self.name == "leader":
            c_combined_logits =  torch.cat([x.view(batch_size, 1, -1), leader_action.reshape(batch_size, 1, self.leader_action_dim).float(), 
                                  follower_action.reshape(batch_size, 1, self.follower_action_dim).float()], -1
                                  )
            action_value = self.linear_critic_1(c_combined_logits.view(batch_size, 1, -1))#F.relu()
            # print("x.size()-------------------------",x.size())
            # action_value = self.linear_critic_2(x)

        return selected_log_prob, action_value.reshape(batch_size,1,1), action, h_state.detach().data, entropy 