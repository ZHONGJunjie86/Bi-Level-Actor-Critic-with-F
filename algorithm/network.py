import torch.cuda
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self):  #(n+2p-f)/s + 1 
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(4,16, kernel_size=3, stride=1, padding=1) # 30 -> 15
        self.maxp1 = nn.MaxPool2d(4, stride=2 , padding= 1)
        self.conv2 = nn.Conv2d(16,16, kernel_size=4, stride=1, padding=0) # 15 -> 12
        self.conv3 = nn.Conv2d(16,8, kernel_size=4, stride=1, padding=0) # 12 -> 9
        self.self_attention = nn.MultiheadAttention(654, 3)

        self.gru = nn.GRU(654, 64, 1) 
        self.critic_linear = nn.Linear(64, 1)
        self.linear_mu = nn.Linear(64, 2)
        self.linear_sigma = nn.Linear(64, 2)

        self.normal =  torch.distributions.Normal
        self.num_vector_length = 6

    def forward(self, tensor_cv, num_vector, h_old,  old_action = None): #,batch_size
        # CV
        self.batch_size = tensor_cv.size()[0]
        i_1 = tensor_cv
        # CV
        x = F.relu(self.maxp1(self.conv1(i_1)))
        #i_2 = i_1 + x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)).reshape(1,self.batch_size,648)#(1,self.batch_size,640)
        
        step = num_vector.reshape(1,self.batch_size,self.num_vector_length)

        x = torch.cat([x, step], -1)
        x = self.self_attention(x,x,x)[0] + x
        x,h_state = self.gru(x, h_old)
        
        value = self.critic_linear(x)

        #[Box(-100.0, 200.0, (1,), float32), Box(-30.0, 30.0, (1,), float32)]
        mu = torch.tanh(self.linear_mu(x)).reshape(self.batch_size, 2, 1)
        sigma = torch.relu(self.linear_sigma(x)).reshape(self.batch_size, 2, 1) + 1e-6

        dist = self.normal(mu,sigma)  
        entropy = dist.entropy().mean()

        if old_action == None:
            action = dist.sample().reshape(self.batch_size, 2,1)
        else: 
            action = old_action.reshape(self.batch_size, 2,1)

        selected_log_prob = dist.log_prob(action)   
        
        return action, selected_log_prob, entropy, h_state.data, value.reshape(self.batch_size,1,1)