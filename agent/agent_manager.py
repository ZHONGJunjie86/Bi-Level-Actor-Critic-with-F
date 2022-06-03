from algorithm.DPPO import PPO


class Agents2Env:
    def __init__(self, agent_name_list, obs_shape, device):
        self.agents = {}
        self.adversaries = {}
        self.agent_name_list = agent_name_list
        self.construct_agent(obs_shape)
        self.device = device

    def construct_agent(self, obs_shape):
        # ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']
        for name in self.agent_name_list:
            if "adversar" in name:
                self.adversaries[name] = PPO(obs_shape[name], self.device)
            else:
                self.agents[name] = PPO(obs_shape[name], self.device)

    def get_actions(self):
        pass

