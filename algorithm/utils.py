class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.hidden_states = []
        self.num_vectors = []
        self.values = []
        self.action_values = []
        self.leader_action_behaviour = []
        self.follower_share_inform = []
    
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden_states[:]
        del self.num_vectors[:]
        del self.values[:]
        del self.action_values[:]
        del self.leader_action_behaviour[:] 
        del self.follower_share_inform [:]

