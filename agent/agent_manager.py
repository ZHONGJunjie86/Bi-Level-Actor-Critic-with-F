from torch import ctc_loss
from algorithm.DPPO import PPO
import copy 
import collections

    
    
class Agents2Env:
    def __init__(self, agent_name_list, obs_shape, 
                       device, main_device, 
                       model_load_path, model_save_path, 
                       args):

        self.agents = {}
        self.adversaries = {}
        self.agent_name_list = agent_name_list
        self.device = device
        self.main_device = main_device
        self.model_load_path = model_load_path
        self.model_save_path = model_save_path
        self.args = args

        self.construct_agent(obs_shape)

    def construct_agent(self, obs_shape_by_type):
        # ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']
        for name in self.agent_name_list:
            if "adversar" in name:
                self.agents[name] = copy.deepcopy(PPO(self.args, 
                                                    obs_shape_by_type["adversary"], self.device, 
                                                    "adversary"))
            else:
                self.agents[name] = copy.deepcopy(PPO(self.args, obs_shape_by_type["agent"], 
                                                      self.device, "agent"))

    def get_action(self, state, reward, done, agent_name):
        return self.agents[agent_name].choose_action(state, reward, done)

    def load_model(self):
        for agent in self.agents.values():
            agent.load_model(self.model_load_path)

    def quick_load_model(self, model_dic):
        for agent in self.agents.values():
            agent.quick_load_model(model_dic) # must shallow copy
                

    def save_model(self):
        self.agents['agent_0'].save_model(self.model_save_path)
        self.agents['adversary_0'].save_model(self.model_save_path)

    def get_agent_name(self, index):
        return self.agent_name_list[index]

    def reset(self):
        for agent in self.agents.values():
            agent.reset()

    def update(self, grads_dict):
        # self.agents['agent_0'].update(grads_dict)
        self.agents['adversary_0'].update(grads_dict)

    def compute_loss(self, training_time):
        for agent in self.agents.values():
            agent.compute_loss(training_time)

    def add_gradient(self, shared_model_dict):
        for agent in self.agents.values():
            agent.add_gradient(shared_model_dict)

    def get_loss(self):
        loss_dict = {"agent":copy.deepcopy(self.agents['agent_0'].loss_dic),
                     "adversary":copy.deepcopy(self.agents['adversary_0'].loss_dic)}
        return loss_dict

    def reset_loss(self):
        for agent in self.agents.values():
            agent.reset_loss()

    def get_actor(self):
        return {"agent":copy.deepcopy(self.agents['agent_0'].get_actor()),
                "adversary": copy.deepcopy(self.agents['adversary_0'].get_actor())}

    def get_data_dict(self):
        share_data_dict = {"agent":{"old_states":[],"leader_action_behaviour":[]},  
                           "adversary":{"old_states":[],"leader_action_behaviour":[]}}
        for name in ["leader", "follower"]:
            share_data_dict["agent"][name] = {}
            share_data_dict["adversary"][name] = {}
        
        for agent in self.agents.values():
            agent_dict = agent.get_share_data_dict()  
            agent_type = agent.agent_type
            
            share_data_dict[agent_type]["old_states"].extend(agent_dict["old_states"])
            share_data_dict[agent_type]["leader_action_behaviour"].extend(agent_dict["leader_action_behaviour"])
            
            for name in ["leader", "follower"]:
                for key in agent_dict[name].keys():
                    if key not in share_data_dict[agent_type][name]:
                        share_data_dict[agent_type][name][key] = agent_dict[name][key]
                    else:
                        share_data_dict[agent_type][name][key].extend(agent_dict[name][key])

        return copy.deepcopy(share_data_dict)
    
    def update_with_share_data(self, data_dict):       
        # self.agents['agent_0'].compute_grad_with_shared_data(data_dict['agent'])
        self.agents['adversary_0'].compute_grad_with_shared_data(copy.deepcopy(data_dict['adversary']))
    
