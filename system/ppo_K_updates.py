import time
import copy

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