import numpy as np
import collections
# from env.save_log import send_curve_data
from system.ppo_K_updates import K_epochs_PPO_training
from env.utils import information_share
from config.config import *


def step(rank, shared_data, args, device, builder):

    # initialize
    agents = builder.build_agents(device)
    env = builder.build_env()

    print("rank ", rank)
    if rank == 0:
        if args.load_model:
            agents.load_model()
        agents.save_model()
    else:
        agents.load_model()

    RENDER = False#args.render #

    total_step_reward = collections.defaultdict(float)
    episode = 0

    while episode < args.max_episodes:
        agents.load_model()
        step = 0
        states = env.reset()
        dones = {}
        rewards = {}
        for name in agents.agent_name_list:
            dones[name] = False
            rewards[name] = 0
        if RENDER and rank == 0 and episode % 10 == 0:
            env.render()

        while True:
            ################################# collect  action #############################
            actions = {}
            
            # pass hidden states between agents who can see each other
            information_share(states, agents, args)
            
            for agent_name in agents.agents.keys():
                
                reward = rewards[agent_name]
                total_step_reward[agent_name] += reward

                if True not in dones.values():
                    action = agents.get_action(states[agent_name] + 0.01, 
                                               reward, dones[agent_name], agent_name)
                    actions[agent_name] = action
                    
            
            states, rewards, dones, infos = env.step(actions)
            step += 1
            if RENDER and rank == 0 and episode % 10 == 0:
                env.render()
            ################################# env rollout ##########################################
            # ================================== collect data & update ========================================
            if True in dones.values():
                if rank == 0:
                    print("Episode: ", episode)

                loss_dict = K_epochs_PPO_training(rank, args, episode, shared_data, agents)

                # if rank == 0:
                #     send_curve_data(loss_dict, total_step_reward)
                
                # reset
                agents.clear_memory()
                total_step_reward = collections.defaultdict(float)
                episode += 1
                if rank == 0:
                    agents.save_model()
                break





