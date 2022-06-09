import torch
import torch.nn as nn
import torch.multiprocessing as mp
from config.config import agent_type_list
import sys

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


class Shared_Data:
    def __init__(self, model_dict, load_path, save_path, main_device):
        self.shared_lock = mp.Manager().Lock()
        self.event = mp.Event()
        self.shared_count = mp.Value("d", 0)
        self.list_1 = mp.Manager().list()
        self.list_2 = mp.Manager().list()
        self.loss = mp.Manager().dict(
            {"a_loss": self.list_1, "c_loss": self.list_2})

        self.load_path = load_path
        self.save_path = save_path

        self.shared_model = {}
        for agent_type in agent_type_list:
            self.shared_model[agent_type] = {}
            for name in ["leader", "follower"]:
                self.shared_model[agent_type][name] = \
                    Shared_grad_buffers(model_dict[agent_type][name], 
                    main_device, agent_type, name).share_memory()

        self.model_dict = model_dict

        # share training data
        self.share_training_data = {}
        for agent_type in agent_type_list:
            self.share_training_data[agent_type] = {}
            self.share_training_data[agent_type]["old_states"] = mp.Manager().list()
            for name in ["leader", "follower"]:
                self.share_training_data[agent_type][name] = {}
                self.share_training_data[agent_type][name]["old_hiddens"] = mp.Manager().list()
                self.share_training_data[agent_type][name]["old_logprobs"] = mp.Manager().list()
                self.share_training_data[agent_type][name]["advantages"] = mp.Manager().list()
                self.share_training_data[agent_type][name]["target_value"] = mp.Manager().list()

        self.share_training_data = mp.Manager().dict(self.share_training_data)


    def reset(self):
        for agent_type in agent_type_list:
            for name in ["leader", "follower"]:
                self.shared_model[agent_type][name].reset()

    def save(self, new_model_dict):
        for agent_type in agent_type_list:
            for name in ["leader", "follower"]:
                self.model_dict[agent_type][name].load_state_dict(
                new_model_dict[agent_type][name].state_dict())


    def update_share_data(self, dict):
        for agent_type in agent_type_list:
            self.share_training_data[agent_type]["old_states"].extend(dict[agent_type]["old_states"])
            for name in ["leader", "follower"]:
                for list_name in self.share_training_data[agent_type][name].keys():
                    self.share_training_data[agent_type][name][list_name].extend(dict[agent_type][name][list_name])


    def reset_share_data(self):
        for agent_type in agent_type_list:
            for name in ["leader", "follower"]:
                for list_name in self.share_training_data[agent_type][name].keys():
                    del self.share_training_data[agent_type][name][list_name][:]


