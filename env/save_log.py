import re
import wandb
from config.config import agent_type_list


wandb.init(project="Bi-Level-Actor-Critic-with-F", entity="zhongjunjie")
wandb.config = {
"learning_rate": 0.0003,
}

def send_curve_data(loss_dict, reward, total_step_reward):
    send_dic = {"reward": reward, "relative_reward": total_step_reward,
                "agents all reward": 0, "adversaries all reward": 0}
    
    for agent_type in agent_type_list:
        for name in ["leader", "follower"]:
            send_dic[agent_type + " " + name] = loss_dict[agent_type][name]

    for name, reward in total_step_reward.items():
        if "adversar" in name:
            send_dic["adversaries all reward"] += reward
        else:
            send_dic["agents all reward"] += reward
        send_dic[name + " reward"] = reward

    wandb.log(send_dic)
