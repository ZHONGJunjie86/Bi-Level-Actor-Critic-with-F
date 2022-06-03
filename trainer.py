from system.parallel import parallel_trainer
from agent.builder import Bulider


def trainer():
    parallel_trainer(Bulider)  

if __name__ == "__main__":
    trainer()