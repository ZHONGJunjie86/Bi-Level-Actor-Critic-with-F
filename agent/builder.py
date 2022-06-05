from config.config import *
from agent.agent_manager import Agents2Env
from algorithm.network import ActorCritic

class Bulider():
    @staticmethod
    def get_agent_names():
        return agent_name_list

    @staticmethod
    def get_args():
        return args

    @staticmethod
    def get_model_load_path():
        return model_load_path

    @staticmethod
    def get_model_save_path():
        return model_save_path
        
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
            model_dict[agent_type] = {}
            for name in ["leader", "follower"]:
                model_dict[agent_type][name] = ActorCritic(obs_shape_by_type[agent_type], 
                                                           action_dim_by_type[name], 
                                                           action_dim_by_type[name], 
                                                           name) 
        return model_dict

    @staticmethod
    def build_env():
        return simple_tag_v2.parallel_env(num_good=args.num_good, num_adversaries=args.num_adversaries,
                                 num_obstacles=args.num_obstacles, max_cycles=args.max_cycles, continuous_actions=False)