import wandb

def send_curve_data(loss_dict, total_step_reward, agent_type_list):
    
    send_dic = {"relative_reward": sum(total_step_reward.values()),
                "agents all reward": 0, "adversaries all reward": 0}

    
    for agent_type in agent_type_list:
        for loss_name in ["a_loss", "c_loss"]:
            for name in ["leader", "follower"]:     
                send_dic[loss_name + " " + agent_type + " " + name] = loss_dict[agent_type][loss_name][name]

    for name, reward in total_step_reward.items():
        if "adversar" in name:
            send_dic["adversaries all reward"] += reward
        else:
            send_dic["agents all reward"] += reward

    wandb.log(send_dic)