def step(rank, shared_data, args, bulider):

    
    print("rank ", rank)
    print(f'device: {device}')

    
    ctrl_agent_index = int(args.controlled_player)
    
    model = PPO(args, device)
    #model_enemy = PPO(args, device)

    model.actor.load_state_dict(
        shared_model.get_actor().state_dict())  # sync with shared model
    #model_enemy.actor.load_state_dict(shared_model.get_actor().state_dict())

    history_success = []
    history_enemy = {}
    RENDER = True#args.render #

    part_accumulated_reward = 0
    total_step_reward = [0,0]
    c_loss, a_loss = 0, 0
    step_reward = 0

    memory = Memory()
    memory_enemy = Memory()


    episode = 0
    episode_enemy_update = 0
    success = 0
    select_pre = False
    distance_dict = {"our_turn":dict(),"enemy":dict()}


    our_turn = False
    # if rank == 0:
    #     wandb.config = {
    #         "learning_rate": 0.0004,
    #     }
        
    #     wandb.init(project="Curling", entity="zhongjunjie")

    while episode < args.max_episodes:

        step = 0
        Gt = 0
        state = env.reset()
        if RENDER and rank == 0 and episode % 10 == 0:
            env.env_core.render()

        obs = np.array(state[ctrl_agent_index]['obs'])/10
        obs_enemy = np.array(state[1-ctrl_agent_index]['obs'])/10

        for _ in range(Memory_size):
            if np.sum(obs) != -90:
                memory.m_obs.append(obs)
                our_turn = True

            if np.sum(obs_enemy) != -90:
                memory_enemy.m_obs.append(obs_enemy)
                our_turn = False

        if our_turn:
            obs = np.stack(memory.m_obs)
        else:obs = np.stack(memory_enemy.m_obs)

        while True:
            pre_ball_nums = sum(state[ctrl_agent_index]["throws left"])
            if np.sum(np.array(state[ctrl_agent_index]['obs'])/10) != -90:
                our_turn = True
            else: our_turn = False

            #TODO another features maybe 
            num_vector = np.array([
                                int(state[ctrl_agent_index]['release']), 
                                state[ctrl_agent_index]["throws left"][0]/10,
                                state[ctrl_agent_index]["throws left"][1]/10,
                                int(our_turn),
                                int(our_turn),
                                count_down/100
                                ])
            ################################# collect  action #############################
            if our_turn:
                action_opponent = [[1], [1]]
            else:
                action_ctrl = [[1], [1]]
            action_raw = model.choose_action(obs, num_vector, our_turn, state[ctrl_agent_index]['release'])
            #action_raw = actions_map[action_raw]
            if our_turn:
                action_ctrl = linear_transformer(action_raw)
                #action_ctrl = [[action_raw[0]], [action_raw[1]]]   
            else:
                action_opponent = linear_transformer(action_raw)#[[action_raw[0]], [action_raw[1]]]   

            action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl, action_opponent]
            
            # if episode>1:
            #     print("our_turn",our_turn,"action--------",action)
            ################################# env rollout ##########################################
            #self.all_observes, reward, self.done, info_before, info_after
            positions = env.env_core.agent_pos
            next_state, reward, done, _, info = env.step(action)

            next_obs = np.array(next_state[ctrl_agent_index]['obs'])/10
            next_obs_enemy = np.array(next_state[1-ctrl_agent_index]['obs'])/10

            # stack Memory
            if np.sum(next_obs) != -90:
                if len(memory.m_obs_next) == 0:
                    if len(memory.m_obs) == 0:
                        for _ in range(Memory_size):
                            memory.m_obs.append(next_obs)
                        memory.m_obs_next = copy.deepcopy(memory.m_obs)
                    else:
                        memory.m_obs_next = copy.deepcopy(memory.m_obs)
                        memory.m_obs_next[-1] = next_obs
                else:
                    del memory.m_obs_next[:1]
                    memory.m_obs_next.append(next_obs)
                    
                next_obs = np.stack(memory.m_obs_next)

            if np.sum(next_obs_enemy) != -90:
                if len(memory_enemy.m_obs_next) == 0:
                    if len(memory_enemy.m_obs) == 0:
                        for _ in range(Memory_size):
                            memory_enemy.m_obs.append(next_obs_enemy)
                        memory_enemy.m_obs_next = copy.deepcopy(memory_enemy.m_obs)
                    else:
                        memory_enemy.m_obs_next = copy.deepcopy(memory_enemy.m_obs)
                        memory_enemy.m_obs_next[-1] = next_obs_enemy
                else:
                    del memory_enemy.m_obs_next[:1]
                    memory_enemy.m_obs_next.append(next_obs_enemy)

                next_obs = np.stack(memory_enemy.m_obs_next)
            
            
            step += 1

            # ================================== reward shaping ========================================
            #done reward [0.0, 100.0]   post_reward[-100.0, 100.0]           
            #reward投完暂胜1 一局完10.0 结束100
            #350.0 距离开始，靠近环90左右,left and right nodes of red line is 200

            step_reward = compute_reward(state, ctrl_agent_index, positions,  distance_dict, our_turn, count_down, step_reward)
            print("step_reward",step_reward,"total_step_reward",total_step_reward)

            count_down -= 1

            if our_turn:
                reward_index = 0
            else:
                reward_index = 1
            
            #只看距离感觉last奖励             
            #part_accumulated_reward = step_reward
            
            changed = False
            if np.sum(np.array(next_state[ctrl_agent_index]['obs']))/10 == -90 or\
                np.sum(np.array(state[ctrl_agent_index]['obs']))/10 == -90:
                if np.sum(np.array(next_state[ctrl_agent_index]['obs']))/10 != -90 or\
                np.sum(np.array(state[ctrl_agent_index]['obs']))/10 != -90:
                    changed = True
            #投完奖励
            if sum(reward) == 1 or sum(reward) == 10 or sum(reward) == 100 or changed:
                # if sum(reward) == 100:
                #     reward[reward.index(100)] = 10
                # winner_index = reward.index(1) if sum(reward) == 1 else reward.index(10) 
                # # our index 1
                # if len(model.memory_our_enemy[winner_index].rewards) != 0:
                #     model.memory_our_enemy[1-winner_index].rewards[-1] +=  0#max (0.5 * sum(reward) ,2)
                #     total_step_reward[1-winner_index] +=  0#max (0.5 * sum(reward) ,2)
                #     model.memory_our_enemy[winner_index].rewards[-1] -=  0#max (0.5 * sum(reward) ,2)
                #     total_step_reward[winner_index] -=  0#max (0.5 * sum(reward) ,2)

                model.memory_our_enemy[reward_index].is_terminals.append(0)
                count_down = 100
                
                total_step_reward[reward_index] += step_reward
                model.memory_our_enemy[reward_index].rewards.append(step_reward)
                part_accumulated_reward = 0

            else: 
                if state[ctrl_agent_index]['release'] == False:
                    model.memory_our_enemy[reward_index].rewards.append(0)
                    total_step_reward[reward_index] += step_reward
                else:
                    model.memory_our_enemy[reward_index].rewards.append(step_reward)
                model.memory_our_enemy[reward_index].is_terminals.append(0)

            #球投完
            # if sum(reward) == 10:
            #     if reward[0] != reward[1]:
            #         post_reward = [reward[0]-100, reward[1]] if reward[0]<reward[1] else [reward[0], reward[1]-100]
            #     else:
            #         post_reward=[-1., -1.]
            

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

                # if rank == 0:
                #     wandb.log({"a_loss": a_loss, "c_loss": c_loss, 
                #     "total_step_reward_0": total_step_reward[0],
                #     "total_step_reward_1": total_step_reward[1],
                #     "our_turn_distance":compute_group_distance([300, 500], distance_dict["our_turn"], positions),
                #     "enemy_distance":compute_group_distance([300, 500], distance_dict["enemy"], positions) })
                
                distance_dict = {"our_turn":dict(),"enemy":dict()}
                
                    #暂不用自博弈
                    # if len(history_success) >= 200 and sum(history_success[-200:]) > 110:
                    #     episode_enemy_update, select_pre = self_playing_update(model_enemy, model_enemy_path, model, shared_lock, 
                    #                                                             args, episode, episode_enemy_update,
                    #                                                             history_enemy, history_success, select_pre)
                    #     history_success = []

                total_step_reward = [0, 0]
                memory.clear_memory()
                memory_enemy.clear_memory()
                env.reset()
                step_reward = 0
                count_down = 100
                episode += 1
                break





def K_epochs_PPO_training(rank, event, 
                          model_enemy, model_enemy_path, model, shared_model, 
                          shared_count, shared_grad_buffer, shared_lock, 
                          K_epochs, args, episode, run_dir, device ):
    if rank == 0:
        print("---------------------------training!")
        training_time = 0
        #shared_model.copy_memory(model.memory_our_enemy)
        while training_time < K_epochs:
            #
            a_loss, c_loss = model.compute_GAE(training_time)

            while shared_count.value < args.processes-1:
                time.sleep(0.01)
            time.sleep(0.01)
            #
            shared_lock.acquire()
            
            model.add_gradient(shared_grad_buffer)

            shared_count.value = 0
            shared_lock.release()
            #
            shared_model.update(copy.deepcopy(shared_grad_buffer.grads), args.processes)
            shared_grad_buffer.reset()

            c_loss, a_loss = model.get_loss()
            model.actor.load_state_dict(
                shared_model.get_actor().state_dict())

            event.set()
            event.clear()
            training_time += 1

        # torch.save(model_enemy.actor.state_dict(), model_enemy_path)
        model.reset_loss()
        shared_model.clear_memory()
        if episode % 20 == 0:
            shared_model.save_model(run_dir, episode)
        return a_loss, c_loss

    else:
        training_time = 0
        while training_time < K_epochs:
            a_loss, c_loss = model.compute_GAE(training_time)

            shared_lock.acquire()

            model.add_gradient(shared_grad_buffer)

            shared_count.value += 1
            shared_lock.release()

            event.wait()

            model.actor.load_state_dict(
                shared_model.get_actor().state_dict())

            training_time += 1

        # enemy_temp = torch.load(model_enemy_path , map_location = device)
        # model_enemy.load_state_dict(enemy_temp)
        return 0, 0