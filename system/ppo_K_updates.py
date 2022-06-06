import time
import copy

def K_epochs_PPO_training(rank, args, episode, shared_data, agents):
    if rank == 0:
        # main processing
        print("---------------------------training!")
        training_time = 0

        while training_time < args.K_epochs:
            # wait
            agents.compute_loss(training_time)

            while shared_data.shared_count.value < args.processes-1:
                time.sleep(0.01)
            time.sleep(0.01)

            # add
            shared_data.shared_lock.acquire()
            agents.add_gradient(shared_data.shared_model)
            shared_data.shared_count.value = 0
            shared_data.shared_lock.release()

            # update
            agents.update(copy.deepcopy(shared_data.shared_model))
            shared_data.reset()
            loss_dict = agents.get_loss()
            agents.save_model()

            shared_data.event.set()
            shared_data.event.clear()
            training_time += 1

        # return data 
        agents.reset_loss()
        agents.clear_memory()
        return loss_dict

    else:
        # workers
        training_time = 0
        while training_time < args.K_epochs:
            agents.compute_loss(training_time)

            # add
            shared_data.acquire()
            agents.add_gradient(shared_data.shared_model)
            shared_data.shared_count.value += 1
            shared_data.shared_lock.release()
            
            # wait
            shared_data.event.wait()

            # load new model
            agents.load_model()

            training_time += 1

        return 0, 0