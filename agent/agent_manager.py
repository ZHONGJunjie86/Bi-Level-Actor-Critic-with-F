from algorithm.DPPO import PPO


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

    def construct_agent(self, obs_shape):
        # ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']
        for name in self.agent_name_list:
            if "adversar" in name:
                self.adversaries[name] = PPO(self.args, obs_shape[name], self.device)
            else:
                self.agents[name] = PPO(self.args, obs_shape[name], self.device)

    def get_action(self, state, reward, done, agent_index):
        name = self.get_agent_name(agent_index)
        return self.agents[name].choose_ation(state, reward, done)

    def load_model(self):
        pass

    def save_model(self):
        pass
    # model.actor.load_state_dict(
    #             shared_model.get_actor().state_dict())

    def get_agent_name(self, index):
        return self.agent_name_list[index]

    def clear_memory(self):
        pass

    def update(self, grads, processes):
        pass

    def compute_loss(self, training_time):
        pass

    def add_gradient(self, shared_model):
        pass

    def reset_loss(self):
        pass
    

