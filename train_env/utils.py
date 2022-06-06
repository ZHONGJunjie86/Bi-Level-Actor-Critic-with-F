
def compute_dis(my_pos, other_pos):
    return (my_pos[0] - other_pos[0])**2 + (my_pos[1] - other_pos[1])**2

def information_share(sta, agents, args):
    # 只和最近的互换
    states = [list(i) for i in sta.values()]

    # adversary
    for i in range(args.num_adversaries):
        my_pos = states[i][2:4]
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
        
        
        agents.agents["adversary_" + str(i)].memory["follower"].hidden_states[-1] = \
                 agents.agents["adversary_" + str(min_adversary_index_position[0])].memory["leader"].hidden_states[-1]

    # agents
    for i in range(args.num_good):
        my_pos = states[i + args.num_adversaries][2:4]
        min_agent_index_position =  [-1, float("inf")]
        other_agent_index = 0

        
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
            agents.agents["agent_" + str(i)].memory["follower"].hidden_states[-1] = \
                    agents.agents["agent_" + str(min_agent_index_position[0])].memory["leader"].hidden_states[-1]