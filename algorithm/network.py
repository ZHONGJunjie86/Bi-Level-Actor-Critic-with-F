import torch.cuda
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, obs_size, leader_action_dim, follower_action_dim, name):  #(n+2p-f)/s + 1 
        super(ActorCritic, self).__init__()
        # share info
        self.self_attention = nn.MultiheadAttention(64, 4)        
        self.gru = nn.GRU(64, 64, 1) 
        self.name = name
        self.leader_action_dim = leader_action_dim

        # actor
        if name == "follower":
            # state 包含 leader 的动作
            self.linear1 = nn.Linear(obs_size + leader_action_dim, 64)
            # actor beta distribution
            self.linear_alpha = nn.Linear(64, follower_action_dim)
            self.linear_beta = nn.Linear(64, follower_action_dim)
            self.beta_dis =  torch.distributions.beta.Beta
            # critic 含两个动作，follower是后算出来的
            self.linear_critic_1 = nn.Linear(64 + follower_action_dim, 64)
        elif name == "leader":
            # state 包含 follower的动作，自己的 one-hot（这俩决定SE）
            self.linear1 = nn.Linear(obs_size + follower_action_dim * 2 + leader_action_dim, 64)
            # actor Categorical
            self.linear_agent = nn.Linear(64, leader_action_dim)
            self.categorical_dis = torch.distributions.Categorical
            # critic 含两个动作，一开始就有
            self.linear_critic_1 = nn.Linear(64, 64)

        # critic == Q
        self.linear_critic_2 = nn.Linear(64, 1)
        

    def forward(self, obs, h_old,  leader_action, follower_action): 
        # share info
        batch_size = obs.size()[0]
        if self.name == "follower":
            obs = torch.cat([obs, leader_action], -1).view(batch_size, 1, -1)
        else:
            obs = torch.cat([obs, leader_action, follower_action], -1).view(batch_size, 1, -1)
        x = torch.relu(self.linear1(obs))
        x = self.self_attention(x,x,x)[0] + x
        x,h_state = self.gru(x, h_old)

        # actor
        if self.name == "follower":
            alpha = torch.relu(self.linear_alpha(x)) + 0.01  # >0
            beta = torch.relu(self.linear_beta(x)) + 0.01  # >0
            dis =  self.beta_dis(alpha.reshape(batch_size,1), beta.reshape(batch_size,1)).sample() 
            action = dis.sample()
            entropy = dis.entropy().mean()
            selected_log_prob = dis.log_prob(action)
            action += 0.5  # 0 for social reward
        else:
            logits = torch.relu(self.linear_agent(x))
            dis =  self.categorical_dis(logits.reshape(batch_size, self.leader_action_dim))
            action = dis.sample()
            entropy = dis.entropy().mean()
            selected_log_prob = dis.log_prob(action)

        # critic == Q
        if self.name == "follower":
            add_follower_act_logits = torch.cat([
                                                 x.view(batch_size, -1), 
                                                 alpha.view(batch_size,-1), 
                                                 beta.view(batch_size,-1)
                                                 ], -1).view(batch_size, 1, -1)
            x = torch.relu(self.linear_critic_1(add_follower_act_logits))
            action_value = self.linear_critic_2(x)
        else:
            x = torch.relu(self.linear_critic_1(x.view(batch_size, -1)))
            action_value = self.linear_critic_2(x)

        return selected_log_prob, action_value.reshape(self.batch_size,1,1), action, h_state.data, entropy 