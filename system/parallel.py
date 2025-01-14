import torch.multiprocessing as mp
import sys
from system.utils import *
import os
from train_env.step import step

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ["PYOPENGL_PLATFORM"] = "egl"

def parallel_trainer(bulider):
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')  # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        # or else you get a deadlock in conv2d
        raise "Must be using Python 3 with linux!"

    main_device = bulider.get_main_device()
    device = bulider.get_device()

    shared_data = Shared_Data(bulider.build_model_dict(),
                               bulider.get_model_load_path(), 
                               bulider.get_model_save_path(),
                               main_device)
    
    bulider_args = bulider.get_args()

    processes = []
    for rank in range(bulider.get_args().processes):  # rank 编号

        if  rank==0:  #rank < bulider.get_args().processes//5 or
            print("start", rank)
            p = mp.Process(target=step, args=(rank, shared_data, bulider_args, main_device,
                                              bulider))
        else:
            p = mp.Process(target=step, args=(rank, shared_data, bulider_args, device,
                                              bulider))

        p.start()
        processes.append(p)
    for p in processes: 
        p.join()