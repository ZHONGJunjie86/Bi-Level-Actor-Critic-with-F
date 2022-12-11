import math
min_dis = 0.8 # max_all_adv_reward = 0.15
max_reward = 0.02

def compute_dis(my_pos, other_pos):
    return other_pos[0]**2 + other_pos[1]**2# (my_pos[0] - other_pos[0])**2 + (my_pos[1] - other_pos[1])**2


def compute_dis_reward(distance_agent_dict, distance_reward_dict, agent_pos):
    for adv_name, pos in distance_agent_dict.items():
        dis = compute_dis(pos,  agent_pos)
        if dis < min_dis:
            distance_reward_dict[adv_name] = min(1/dis/1e5, max_reward)



def information_share(sta, rewards, agents, args):
    # type reward
    type_reward_dict = {"adversary":0, "agent":0}
    for name in rewards.keys():
        if "adversary" in name:
            type_reward_dict["adversary"] += rewards[name]
        else:
            type_reward_dict["agent"] += rewards[name]
    
    
    
    # 只和最近的互换
    states = [list(i) for i in sta.values()]
    distance_reward_dict = {}
    distance_agent_dict = {}
    
    
    # adversary
    for i in range(args.num_adversaries):
        my_pos = states[i][2:4]
        distance_agent_dict[agents.get_agent_name(i)] = my_pos
        distance_reward_dict[agents.get_agent_name(i)] = 0
        min_adversary_index_position = [-1, float("inf")]
        other_adversary_index = 0

        start_point = 4 + args.num_obstacles * 2
        for j in range(args.num_adversaries): 
            if j == i:
                other_adversary_index += 1
                continue
            dis = compute_dis(my_pos, states[i][start_point: start_point+2])
            if dis < min_adversary_index_position[1]:
                min_adversary_index_position = [other_adversary_index, dis]
            
            start_point += 2
            other_adversary_index += 1
        
        if min_adversary_index_position[0] != -1:
            agents.agents["adversary_" + str(i)].memory["follower"].follower_share_inform.append(
                    agents.agents["adversary_" + str(min_adversary_index_position[0])].memory["leader"].hidden_states[-1])

    # agents
    for i in range(args.num_good):
        my_pos = states[i + args.num_adversaries][2:4]
        min_agent_index_position =  [-1, float("inf")]
        other_agent_index = 0
        compute_dis_reward(distance_agent_dict, distance_reward_dict, my_pos)

        
        start_point = 4 + args.num_obstacles * 2 + args.num_adversaries * 2
        for j in range(args.num_good): 
            if j == i:
                other_agent_index += 1
                continue
            dis = compute_dis(my_pos, states[i + args.num_adversaries][start_point: start_point+2])
            if dis < min_agent_index_position[1]:
                min_agent_index_position = [other_agent_index, dis]
            
            start_point += 2
            other_agent_index += 1
        
        if min_agent_index_position[0] != -1:
            agents.agents["agent_" + str(i)].memory["follower"].follower_share_inform.append(agents.agents["agent_" + str(i)].hidden_state_zero.numpy())
            # (
            #         agents.agents["agent_" + str(min_agent_index_position[0])].memory["leader"].hidden_states[-1])
        else:
            agents.agents["agent_" + str(i)].memory["follower"].follower_share_inform.append(agents.agents["agent_" + str(i)].hidden_state_zero.numpy())
    return distance_reward_dict, type_reward_dict