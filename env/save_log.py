import re
import wandb

wandb.init(project="my-test-project", entity="zhongjunjie")
wandb.config = {
"learning_rate": 0.0003,
}

def send_curve_data(a_loss, c_loss, reward, total_step_reward):
    send_dic = {"a_loss": a_loss, "c_loss": c_loss,
                "reward": reward, "relative_reward": total_step_reward,
                "agents all reward": 0, "adversaries all reward": 0}
    
    for name, reward in total_step_reward.items():
        if "adversar" in name:
            send_dic["adversaries all reward"] += reward
        else:
            send_dic["agents all reward"] += reward
        send_dic[name + " reward"] = reward

    wandb.log(send_dic)
