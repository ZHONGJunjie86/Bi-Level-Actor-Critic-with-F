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
            
            shared_data.shared_lock.acquire()
            if args.share_grad == 1:  # use add gradient
                # add
                agents.add_gradient(shared_data.shared_model)
                # update
                agents.update(copy.deepcopy(shared_data.shared_model))
                shared_data.reset()
                shared_data.save(agents.get_actor())
            else:    # use share data
                # add
                shared_data.update_share_data(agents.get_data_dict())#copy.deepcopy()
                # update
                # print("---------------------------center training!") 
                loss_dict = shared_data.train()
                # print("---------------------------center training over!")
                # agents.update_with_share_data(copy.deepcopy(shared_data.share_training_data))
                shared_data.reset_share_data()
                training_time = args.K_epochs
            
            shared_data.shared_count.value = 0
            agents.quick_load_model(shared_data.model_dict) # must shallow copy
            shared_data.shared_lock.release()
            # print("---------------------------center loading over!")
            
            
            shared_data.event.set()
            shared_data.event.clear()
            training_time += 1

        # return data 
        if args.share_grad == 1:
            loss_dict = agents.get_loss()
        return loss_dict

    else:
        # workers
        training_time = 0
        while training_time < args.K_epochs:
            # print("---------------------------worker computing!")
            agents.compute_loss(training_time)
            # print("---------------------------worker computing over!")
            # add
            if args.share_grad == 1:
                shared_data.shared_lock.acquire()
                agents.add_gradient(shared_data.shared_model)
            else:
                shared_data.shared_lock.acquire()
                shared_data.update_share_data(agents.get_data_dict())#copy.deepcopy()
                training_time = args.K_epochs
                # print("---------------------------worker updating over!")
            shared_data.shared_count.value += 1
            shared_data.shared_lock.release()
            
            # wait
            shared_data.event.wait()

            # load new model
            shared_data.shared_lock.acquire()
            agents.quick_load_model(shared_data.model_dict) # must shallow copy
            if args.share_grad == 0:
                loss_dict = shared_data.get_loss_dict()
            shared_data.shared_lock.release()
            # print("---------------------------worker loading over!")

            training_time += 1
        return loss_dict if args.share_grad == 0 else {}