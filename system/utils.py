import torch
import torch.multiprocessing as mp


class Shared_grad_buffers:
    def __init__(self, models, main_device):
        self.device = main_device
        self.grads = {}
        for name, p in models.named_parameters():
            self.grads[name + '_grad'] = torch.zeros(p.size()).share_memory_().to(self.device)


    def add_gradient(self, models):
        for name, p in models.named_parameters():
            #print("name, p",name,p)
            self.grads[name + '_grad'] += p.grad.data.to(self.device)

    def reset(self):
        for name, grad in self.grads.items():
            self.grads[name].fill_(0)


class Shared_Data:
    def __init__(self, agent_model, adversary_model, load_path, save_path, main_device):
        self.shared_lock = mp.Manager().Lock()
        self.event = mp.Event()
        self.shared_count = mp.Value("waiting_process", 0)
        self.list_1 = mp.Manager().list()
        self.list_2 = mp.Manager().list()
        self.loss = mp.Manager().dict(
            {"a_loss": self.list_1, "c_loss": self.list_2})

        self.load_path = load_path
        self.save_path = save_path
        self.shared_agent_model = Shared_grad_buffers(agent_model, main_device)
        self.shared_adversary_model = Shared_grad_buffers(adversary_model, main_device)

    def save_model(self):
        pass

