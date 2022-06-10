import numpy as np
import collections
from train_env.save_log import send_curve_data
from system.ppo_K_updates import K_epochs_PPO_training
from train_env.utils import information_share
from config.config import *
import wandb
import copy
import time

def step(rank, shared_data, args, device, builder):

    # initialize
    agents = builder.build_agents(device)
    env = builder.build_env()

    
    if rank == 0:
        print("rank ", rank)
        if args.load_model:
            shared_data.load()
        
        agents.quick_load_model(shared_data.model_dict)
        wandb.init(project="Bi-Level-Actor-Critic-with-F", entity="zhongjunjie")
        wandb.config = {
        "learning_rate": 0.0003,
        }  # waiting for all event.wait() and start them
        shared_data.event.set()
        shared_data.event.clear()
        
    else:
        shared_data.event.wait()
        shared_data.shared_lock.acquire()
        agents.quick_load_model(shared_data.model_dict)
        shared_data.shared_lock.release()
        print("rank ", rank)
        
    RENDER = True#args.render #

    total_step_reward = collections.defaultdict(float)
    episode = 0

    while episode < args.max_episodes:
        if rank == 0:
            print("-----------------Episode: ", episode)
        
        step = 0
        states = env.reset()
        dones = {}
        rewards = {}
        for name in agents.agent_name_list:
            dones[name] = False
            rewards[name] = 0
        if RENDER and rank == 0 and episode % 10 == 0:
            env.render()
            time.sleep(0.1)

        while True:
            ################################# collect  action #############################
            actions = {}
            
            # pass hidden states between agents who can see each other
            distance_reward_dict = information_share(states, agents, args)
            
            for agent_name in agents.agents.keys():
                
                if "agent" in agent_name:reward = rewards[agent_name]/100
                else: reward = rewards[agent_name]/10  + distance_reward_dict[agent_name]
                total_step_reward[agent_name] += reward

                if True not in dones.values():
                    action = agents.get_action(states[agent_name], 
                                               reward, dones[agent_name], agent_name)
                    if "agent" in agent_name:
                        actions[agent_name] = 0
                    else:
                        actions[agent_name] = action
                    # actions[agent_name] = action
            
            states, rewards, dones, infos = env.step(actions)
            step += 1
            if RENDER and rank == 0 and episode % 10 == 0:
                env.render()
                time.sleep(0.1)
            ################################# env rollout ##########################################
            # ================================== collect data & update ========================================
            if True in dones.values():
                if rank == 0:
                    print("Episode ", episode, " over ")

                loss_dict = K_epochs_PPO_training(rank, args, episode, shared_data, agents)
                
                if rank == 0:
                    send_curve_data(loss_dict, total_step_reward, agent_type_list)
                
                # reset
                agents.reset()
                total_step_reward = collections.defaultdict(float)
                episode += 1
                if rank == 0:
                    if episode % 10 == 0:
                        shared_data.save()
                    
                break





