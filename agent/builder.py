from config.config import *
from agent_manager import Agents2Env
from algorithm.network import ActorCritic

class Bulider():
    @staticmethod
    def get_agent_names():
        return agent_name_list

    @staticmethod
    def get_args():
        return args

    @staticmethod
    def get_model_load_path(agent_type):
        return model_load_path[agent_type]

    @staticmethod
    def get_model_save_path(agent_type):
        return model_save_path[agent_type]

    @staticmethod
    def get_main_device():
        return main_device

    @staticmethod
    def get_device():
        return device

    @staticmethod
    def build_agents(run_device):
        return Agents2Env(agent_name_list, obs_shape, 
                          run_device, main_device,
                          model_load_path, model_save_path,
                          args) 

    @staticmethod
    def build_model_dict():
        model_dict = {}
        for agent_type in agent_type_list:
            for name in ["leader", "follower"]:
                model_dict[agent_type][name] = ActorCritic(obs_shape_by_type[agent_type], 
                                                           action_dim_by_type[name], 
                                                           action_dim_by_type[name], 
                                                           name) 
        return model_dict

    @staticmethod
    def build_env():
        return simple_tag_v2.parallel_env(args.num_good, 
                args.num_adversaries, args.num_obstacles, 
                args.max_cycles, continuous_actions=False)