import numpy as np
import collections
from save_log import send_curve_data
from system.ppo_K_updates import K_epochs_PPO_training
from utils import information_share


def step(rank, shared_data, args, agents, env):

    # initialize
    print("rank ", rank)
    agents.load_model()

    RENDER = True#args.render #

    total_step_reward = collections.defaultdict(float)
    c_loss, a_loss = 0, 0
    episode = 0

    while episode < args.max_episodes:
        step = 0
        state = env.reset()
        if RENDER and rank == 0 and episode % 10 == 0:
            env.render()

        while True:
            ################################# collect  action #############################
            agent_index = 0
            states = {}
            while agent_index < args.agent_nums:
                
                state, reward, done, info = env.last()
                total_step_reward[agents.get_agent_name(agent_index)] += reward
                states[agents.get_agent_name(agent_index)] = state
                
                # pass hidden states between agents who can see each other
                information_share(states, agents)

                if not done:
                    agents.get_action(state, reward, done, agent_index)
                    action = 0
                    env.step(action)
            
                agent_index += 1
            
            step += 1
            if RENDER and rank == 0 and episode % 10 == 0:
                env.render()
            ################################# env rollout ##########################################
            # ================================== collect data & update ========================================
            if done:
                if rank == 0:
                    print("Episode: ", episode)

                loss_dict = K_epochs_PPO_training(rank, args, episode, shared_data, agents)

                if rank == 0:
                    send_curve_data(loss_dict, total_step_reward)
                
                # reset
                agents.clear_memory()
                total_step_reward = collections.defaultdict(float)
                episode += 1
                break





