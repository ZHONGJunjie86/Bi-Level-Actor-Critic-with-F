from pettingzoo.mpe import simple_tag_v2
import argparse
import datetime
import sys
from pathlib import Path
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', default=0.99, type=float)  # 0.95
parser.add_argument('--a_lr', default=0.0003, type=float)  # 0.0001

parser.add_argument('--render', action='store_true')
parser.add_argument("--save_interval", default=20, type=int)  # 1000
parser.add_argument("--model_episode", default=0, type=int)
parser.add_argument(
    '--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
# env
parser.add_argument("num_good",  default=1, type=int)  
parser.add_argument("num_adversaries",  default=3, type=int)  
parser.add_argument("num_obstacles",  default=2, type=int) 
parser.add_argument("agent_nums", default=4, type=int)
parser.add_argument("max_cycles", default=25, type=int)  # Agent Environment Cycle 等于游戏步

# PPO
parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
parser.add_argument("--load_model_run", default=8, type=int)
parser.add_argument("--load_model_run_episode", default=4000, type=int)
parser.add_argument("--K_epochs", default=5, type=int)

# Multiprocessing
parser.add_argument('--processes', default=1, type=int,
                    help='number of processes to train with')

                                

args = parser.parse_args()


# 环境相关
env = simple_tag_v2.parallel_env(args.num_good, args.num_adversaries, args.num_obstacles, args.max_cycles, continuous_actions=False)
agent_name_list = [agent_name for agent_name in env.agents]
obs_shape = {env.observation_spaces[agent_name].shape[0] for agent_name in env.agents}

# 定义保存路径
model_load_path = {"agent": "", "adversary":""}
model_save_path = {"agent": "", "adversary":""}

# multiprocessing
main_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") 
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu") 




    