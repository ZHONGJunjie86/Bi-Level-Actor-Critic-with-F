import numpy as np
import collections
# from env.save_log import send_curve_data
from system.ppo_K_updates import K_epochs_PPO_training
from env.utils import information_share


def step(rank, shared_data, args, agents, env):

    # initialize
    print("rank ", rank)
    if rank == 0:
        if args.load_model:
            agents.load_model()
        agents.save_model()
    else:
        agents.load_model()

    RENDER = True#args.render #

    total_step_reward = collections.defaultdict(float)
    c_loss, a_loss = 0, 0
    episode = 0

    while episode < args.max_episodes:
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
            agent_index = 0
            actions = {}
            
            # pass hidden states between agents who can see each other
            information_share(states, agents, args)

            while agent_index < args.agent_nums:
                reward = rewards[agents.get_agent_name(agent_index)]
                total_step_reward[agents.get_agent_name(agent_index)] += reward

                if True not in dones.values():
                    action = agents.get_action(states[agents.get_agent_name(agent_index)], 
                                               reward, dones[agents.get_agent_name(agent_index)], agent_index)
                    actions[agents.get_agent_name(agent_index)] = action
                    

                #states[agents.get_agent_name(agent_index)] = state
                agent_index += 1
            
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
                break





