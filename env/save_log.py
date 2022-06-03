import wandb

wandb.init(project="my-test-project", entity="zhongjunjie")
wandb.config = {
"learning_rate": 0.0003,
"batch_size": args.batch_size
}

def send_curve_data(a_loss, c_loss, reward, relative_reward):
    wandb.log({"a_loss": a_loss, "c_loss": c_loss,
                "reward": reward, "relative_reward": relative_reward})


# np.sum(episode_reward[0:3])
# (np.sum(episode_reward[0:3])-np.sum(episode_reward[3:]))