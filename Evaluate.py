import torch, os
import matplotlib.pyplot as plt
import numpy as np
from Module_N_C import N_C
from Module_N_AG import N_AG
from Module_N_DP import N_DP
from Vars import *


def PlotTemplate(fontsize):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['mathtext.fontset'] = 'stix'

def SaveAllActiveFigures(IndexingName: str = "Figure"):
    if not os.path.exists("Figures"):
        os.mkdir("Figures")
    for fignum in plt.get_fignums():
        plt.figure(fignum)
        plt.savefig(f"Figures/{IndexingName}_{fignum:02d}.png", dpi=400, bbox_inches='tight')
        # plt.savefig(f"Figures/{fignum}.eps", format='eps')
        print(f"Figures/{IndexingName}_{fignum:02d}.png Saved.")

def Plot_Loss():
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(Nc_Model['yddot_training_loss_history'], color='black', linewidth=1.5, label='$\mathcal{N}_{C}$')
    ax.plot(Nag_Model['yddot_training_loss_history'], color='blue', linewidth=1, label='$\mathcal{N}_{AG}$')
    ax.plot(Ndp_Model['yddot_training_loss_history'], color='red', linewidth=.7, label='$\mathcal{N}_{DP}$')
    ax.set_ylim(np.power(10, 0.8), np.power(10, 6))
    ax.grid(linewidth=0.5)
    leg = ax.legend(loc=1)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)
    
    ax.set(xlim=(0, 5000), xlabel="Epochs", yscale='log')
    ylabelfontsize = 18
    ax.set_ylabel(f'MSE of unscaled $\\ddot{{y}}$', fontsize=ylabelfontsize)
    ax.set_yticks([1e2, 1e4, 1e6])
    ax.set_yticklabels(["$10^2$", "$10^4$", "$10^6$"], fontsize=ylabelfontsize - 3)
    fig.tight_layout()  # plt.show()

def Plot_Transient():
    plotStyle_data = dict(color='grey', linestyle='--', linewidth=2)
    plotStyle_Conv = dict(color='black', marker=None, linestyle='-', linewidth=2)
    plotStyle_PINN = dict(color='blue', marker=None, linestyle='-', linewidth=1.5)
    plotStyle_DPCNN = dict(color='red', marker=None, linestyle='-', linewidth=1.5)
    
    fig, ax = plt.subplots(3, 3, figsize=(19, 12))
    ax[0, 0].set_title('$\mathcal{N}_{C}$', fontsize=30)
    ax[0, 1].set_title('$\mathcal{N}_{AG}$', fontsize=30)
    ax[0, 2].set_title('$\mathcal{N}_{DP}$', fontsize=30)
    for iii in [0, 1, 2]:
        ax[2, iii].set_xlabel('$t$', fontsize=34)
    # ax[2,1].set_xlabel('$t$',fontsize=30)
    # ax[2,2].set_xlabel('$t$',fontsize=30)
    for i in range(3):
        ax[0, i].plot(train_x, train_y, **plotStyle_data)
        ax[1, i].plot(train_x, train_yDot, **plotStyle_data)
        ax[2, i].plot(train_x, train_yDDot, **plotStyle_data)
    ax[2, 0].plot(train_x, pred_yDDot_Conv, **plotStyle_Conv)
    
    ax[0, 1].plot(train_x, pred_y_PINN, **plotStyle_PINN)
    ax[1, 1].plot(train_x, pred_yDot_PINN, **plotStyle_PINN)
    ax[2, 1].plot(train_x, pred_yDDot_PINN, **plotStyle_PINN)
    
    ax[0, 2].plot(train_x, pred_y_DPCNN, **plotStyle_DPCNN)
    ax[1, 2].plot(train_x, pred_yDot_DPCNN, **plotStyle_DPCNN)
    ax[2, 2].plot(train_x, pred_yDDot_DPCNN, **plotStyle_DPCNN)
    
    yLabels = [f"$y$", f"$\\dot{{y}}$", f"$\\ddot{{y}}$"]
    for j in range(ax.shape[0]):
        for k in range(ax.shape[1]):
            if k == 0:
                ax[j, k].set_ylabel(yLabels[j], fontsize=34)
            
            if j == 0:
                ax[j, k].set(ylim=(-3, 3), yticks=np.arange(-3, 3.1, 1.5))
            elif j == 1:
                ax[j, k].set(ylim=(-20, 20), yticks=np.arange(-20, 21, 10))
            elif j == 2:
                ax[j, k].set(ylim=(-500, 500), yticks=np.arange(-500, 501, 250))
            
            template_Transient(ax[j, k])
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].grid()
            if not i == 2:
                ax[i, j].set_xticks(np.arange(-2.5, 7.6, 2.5))
                ax[i, j].set_xticklabels([])
            else:
                ax[i, j].set_xticks(np.arange(-2.5, 7.6, 2.5))
            
            if not j == 0:
                ax[i, j].set_yticklabels([])
    
    fig.tight_layout()

def template_Transient(ax):
    ax.set(xlim=(-2.5, 7.5), xticks=np.arange(-2.5, 7.6, 2.5))  # old


def TrainDataVisualization():
    data = pd.read_csv("Dataset/TrainData.csv")
    PlotTemplate(15)
    fig, ax = plt.subplots(3, 1, figsize=(13, 8))
    
    axStyle_y = dict(color='grey', marker='o', markerfacecolor='none', markeredgecolor='grey', linewidth=1,
                     linestyle='dashed', markersize=4)
    plotStyle_y = dict(xlim=(-2.5, 7.5), ylim=(-3, 3), xticks=np.arange(-2.5, 7.6, 2.5), yticks=np.arange(-3, 4, 1.5),
                       ylabel='$y(t)$', xticklabels=[])
    plotStyle_dy = dict(xlim=(-2.5, 7.5), ylim=(-20, 20), xticks=np.arange(-2.5, 7.6, 2.5),
                        yticks=np.arange(-20, 21, 10), ylabel='$\dot{y}(t)$', xticklabels=[])
    plotStyle_ddy = dict(xlim=(-2.5, 7.5), ylim=(-500, 500), xticks=np.arange(-2.5, 7.6, 2.5),
                         yticks=np.arange(-500, 501, 250), xlabel='$t$', ylabel='$\ddot{y}(t)$')
    
    plt.rcParams['axes.titlepad'] = 10
    ax[0].plot(data['x'], data['y'], **axStyle_y)
    ax[0].set(**plotStyle_y)
    # ax[0].grid()
    
    ax[1].plot(data['x'], data['yDot'], **axStyle_y)
    ax[1].set(**plotStyle_dy)
    # ax[1].grid()
    
    ax[2].plot(data['x'], data['yDDot'], **axStyle_y)
    ax[2].set(**plotStyle_ddy)
    # ax[2].grid()
    
    ax[0].set_ylabel('$y$', fontsize=25)
    ax[1].set_ylabel('$\dot{y}$', fontsize=25)
    ax[2].set_ylabel('$\ddot{y}$', fontsize=25)
    ax[2].set_xlabel('$t$', fontsize=25)
    
    for i in range(3):
        ax[i].grid()
    
    fig.tight_layout()  # plt.show()


TrainData = pd.read_csv(f"{DatasetPath}\\TrainData.csv")
train_x = torch.FloatTensor(TrainData[t].to_numpy()).reshape(-1, 1)
train_y = torch.FloatTensor(TrainData[y].to_numpy()).reshape(-1, 1)
train_yDot = torch.FloatTensor(TrainData[yDot].to_numpy()).reshape(-1, 1)
train_yDDot = torch.FloatTensor(TrainData[yDDot].to_numpy()).reshape(-1, 1)

# Load state_dict && loss
Nc_Model = torch.load(f"{ModelPath}\\N_C.pt")
Nag_Model = torch.load(f"{ModelPath}\\N_AG.pt")
Ndp_Model = torch.load(f"{ModelPath}\\N_DP.pt")

# Define net on CPU
Nc = N_C()
Nag = N_AG()
Ndp = N_DP()

Nc.load_model_info(Nc_Model)
Nag.load_model_info(Nag_Model)
Ndp.load_model_info(Ndp_Model)

Nc.count_params()
Nag.count_params()
Ndp.count_params()

pred_yDDot_Conv = Nc.forward_with_normalization(train_x)
pred_y_PINN, pred_yDot_PINN, pred_yDDot_PINN = Nag.forward_with_normalization(train_x)
pred_y_DPCNN, pred_yDot_DPCNN, pred_yDDot_DPCNN = Ndp.forward_with_normalization(train_x)

PlotTemplate(18)
TrainDataVisualization()
Plot_Loss()
Plot_Transient()
SaveAllActiveFigures("Figure")
