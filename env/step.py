import numpy as np
from tensorboard import notebook
from save_log import send_curve_data


def step(rank, shared_data, args, agents, env):

    # initialize
    print("rank ", rank)
    agents.load_model()

    RENDER = True#args.render #

    history_success = []
    history_enemy = {}
    total_step_reward = [0,0]
    c_loss, a_loss = 0, 0
    step_reward = 0

    episode = 0
    success = 0

    while episode < args.max_episodes:
        step = 0
        state = env.reset()
        if RENDER and rank == 0 and episode % 10 == 0:
            env.env_core.render()

        while True:
            ################################# collect  action #############################
            step_cnt = 0
            while step_cnt < args.agent_nums:
                agent_idx = step_cnt % 4
                next_state, reward, done, info = env.last()
                if not done:
                    # print('step cnt : {}'.format(step_cnt))
                    print('next_state : {}, reward : {}, done : {}, info : {}'.format(next_state, reward, done, info))
                    action = 0
                    

                    env.step(action)
                    # env.render()
                step_cnt += 1
            

            ################################# env rollout ##########################################
            #self.all_observes, reward, self.done, info_before, info_after
            positions = env.env_core.agent_pos
            next_state, reward, done, _, info = env.step(action)

            next_obs = np.array(next_state[ctrl_agent_index]['obs'])/10
            next_obs_enemy = np.array(next_state[1-ctrl_agent_index]['obs'])/10
            
            
            step += 1

            # ================================== reward shaping ========================================
            step_reward = compute_reward(state, ctrl_agent_index, positions,  distance_dict, our_turn, count_down, step_reward)
            

            if RENDER and rank == 0 and episode % 10 == 0:
                env.env_core.render()

            # ================================== collect data ========================================
            # Store transition in R
            state = next_state
            obs = next_obs
            #obs_enemy = next_obs_enemy
            
            if sum(reward) == 10 or done or sum(reward) == 100 or pre_ball_nums<sum(next_state[ctrl_agent_index]["throws left"]):
                model.memory_our_enemy[0].is_terminals.append(1)
                model.memory_our_enemy[1].is_terminals.append(1)
                
                win_is = 1 if reward[ctrl_agent_index]>reward[1-ctrl_agent_index] else 0
                win_is_op = 1 if reward[ctrl_agent_index]<reward[1-ctrl_agent_index] else 0

                if rank == 0:
                    print("Episode: ", episode)


                #writer.add_scalar('training Gt', Gt, episode)
                # Training
                a_loss, c_loss = K_epochs_PPO_training(rank, event, 
                          None, model_enemy_path, model, shared_model,   #model_enemy 暂不用自博弈
                          shared_count, shared_grad_buffer, shared_lock, 
                          K_epochs, args, episode, run_dir, device )

                model.memory_our_enemy[0].clear_memory()
                model.memory_our_enemy[1].clear_memory()

                if win_is:
                    success = 1
                else:
                    success = 0
                history_success.append(success)

                if rank == 0:
                    send_curve_data()
                
                total_step_reward = [0, 0]
                env.reset()
                step_reward = 0
                count_down = 100
                episode += 1
                break





