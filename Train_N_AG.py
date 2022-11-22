from Vars import *
from Module_N_AG import N_AG

def Main():
    Nag = N_AG(len(t), hlayers_others, neurons_others, len(y)).cuda()
    Nag.count_params()
    Nag.setup_dataloader(TrainData, ValidData, batch_size=global_batchsize, metamodeling=False,
                         pin_memory=global_pin_memory, num_workers=global_num_workers)
    Nag.fit(global_epochs, global_halflife, 'N_AG.pt', save_every=global_save_every, save=True,
            print_every=global_print_every)

if __name__ == '__main__':
    Main()
