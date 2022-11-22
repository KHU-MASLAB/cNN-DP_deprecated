from Vars import *
from Module_N_DP import N_DP

def Main():
    Ndp = N_DP(len(t), hlayers_dp, neurons_dp, len(y)).cuda()
    Ndp.count_params()
    Ndp.setup_dataloader(TrainData, ValidData, batch_size=global_batchsize, metamodeling=False,
                         pin_memory=global_pin_memory, num_workers=global_num_workers)
    Ndp.fit(global_epochs, global_halflife, 'N_DP.pt', save_every=global_save_every, save=True,
            print_every=global_print_every)

if __name__ == '__main__':
    Main()
