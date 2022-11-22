from Vars import *
from Module_N_C import N_C

def Main():
    Nc = N_C(len(t), hlayers_others, neurons_others, len(y)).cuda()
    Nc.count_params()
    Nc.setup_dataloader(TrainData, ValidData, batch_size=global_batchsize, metamodeling=False,
                        pin_memory=global_pin_memory, num_workers=global_num_workers)
    Nc.fit(global_epochs, global_halflife, 'N_C.pt', save_every=global_save_every, save=True,
           print_every=global_print_every)

if __name__ == '__main__':
    Main()
