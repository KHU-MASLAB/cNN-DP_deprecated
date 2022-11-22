import torch, time, warnings, os
import numpy as np
from Module_N_C import N_C, MLP
from sklearn.metrics import r2_score
from Vars import *

class N_DP(N_C):
    def __init__(self,
            in_N: int = 1,
            depth: int = 2,
            width: int = 2,
            out_N: int = 1,
            activation: str = 'gelu',
            param_init: str = 'default',
            batchnorm: bool = False,
            dropout: float = 0):
        super().__init__()
        ################## model info ##################
        vars = locals()
        if 'self' in vars.keys(): del vars['self']
        if '__class__' in vars.keys(): del vars['__class__']
        self.model_info = vars
        self.model_info.update(vars)
        self.model_info = {key: value for key, value in sorted(self.model_info.items())}
        ################## model info ##################
        
        self.N_DP0 = MLP(in_N, depth, width, out_N, activation=activation, param_init=param_init, batchnorm=batchnorm,
                         dropout=dropout)
        self.N_DP1 = MLP(in_N + out_N, depth, width, out_N, activation=activation, param_init=param_init,
                         batchnorm=batchnorm, dropout=dropout)
        self.N_DP2 = MLP(in_N + 2 * out_N, depth, width, out_N, activation=activation, param_init=param_init,
                         batchnorm=batchnorm, dropout=dropout)
    
    def forward(self, x):
        y = self.N_DP0.forward(x)
        yDot = self.N_DP1.forward(torch.cat((x, y), dim=1))
        yDDot = self.N_DP2.forward(torch.cat((x, y, yDot), dim=1))
        return y, yDot, yDDot
    
    def setup_optimizer(self, initial_lr: float = 1e-3, weight_decay: float = 0, lr_set: list = None):
        ################## model info ##################
        vars = locals()
        if 'self' in vars.keys(): del vars['self']
        if '__class__' in vars.keys(): del vars['__class__']
        self.model_info.update(vars)
        self.model_info = {key: value for key, value in sorted(self.model_info.items())}
        ################## model info ##################
        
        if not lr_set:
            dict1 = {'params': self.N_DP0.parameters(), 'lr': initial_lr, 'weight_decay': weight_decay}
            dict2 = {'params': self.N_DP1.parameters(), 'lr': initial_lr, 'weight_decay': weight_decay}
            dict3 = {'params': self.N_DP2.parameters(), 'lr': initial_lr, 'weight_decay': weight_decay}
        elif lr_set:
            dict1 = {'params': self.N_DP0.parameters(), 'lr': lr_set[0], 'weight_decay': weight_decay}
            dict2 = {'params': self.N_DP1.parameters(), 'lr': lr_set[1], 'weight_decay': weight_decay}
            dict3 = {'params': self.N_DP2.parameters(), 'lr': lr_set[2], 'weight_decay': weight_decay}
        
        # self.optimizer1 = torch.optim.RAdam([dict1, dict2, dict3])
        self.optimizer1 = torch.optim.RAdam([dict1])
        self.optimizer2 = torch.optim.RAdam([dict2])
        self.optimizer3 = torch.optim.RAdam([dict3])
    
    def fit(self,
            epochs,
            lr_halflife,
            filename,
            save: bool = True,
            save_every: int = 1,
            reg_lambda: float = 0,
            initial_lr: float = 1e-3,
            loss_fn: str = 'mse',
            print_every: int = 10,
            weight_decay: float = 0,
            lr_set: list = None,
            valid_measure: str = 'mse'):
        ################## model info ##################
        vars = locals()
        if 'self' in vars.keys(): del vars['self']
        if '__class__' in vars.keys(): del vars['__class__']
        self.model_info.update(vars)
        self.model_info = {key: value for key, value in sorted(self.model_info.items())}
        ################## model info ##################
        
        self.setup_optimizer(initial_lr=initial_lr, weight_decay=weight_decay, lr_set=lr_set)
        self.setup_loss_fn(loss_fn)
        
        for key, value in self.model_info.items():
            print(f"{key.upper()}: {value}")
        print()
        
        for module in self.children():
            print(module)
        print()
        
        tLoss_yDDot_history = np.empty(epochs)
        vLoss_yDDot_history = np.empty(epochs)
        model_parameter_group = np.empty(epochs, dtype=object)
        model_validation_loss = np.full(epochs, 1e30)
        TrainingStartTime = time.time()
        
        for epoch in range(self.model_info['epochs']):
            
            # Training step
            tLabel = []
            tPrediction = []
            epoch_computation_time = 0
            for trainbatch_idx, trainbatch in enumerate(self.train_dataloader):
                batch_start = time.time()
                # Forward
                if self.model_info['metamodeling']:
                    train_x, train_params, train_y, train_yDot, train_yDDot = trainbatch
                    train_x = train_x.cuda()
                    train_params = train_params.cuda()
                    train_y = train_y.cuda()
                    train_yDot = train_yDot.cuda()
                    train_yDDot = train_yDDot.cuda()
                    
                    with torch.no_grad():
                        train_x = (train_x - self.mean_x) / self.std_x
                        train_params = (train_params - self.mean_params) / self.std_params
                        train_y = (train_y - self.mean_y) / self.std_y
                        train_yDot = (train_yDot - self.mean_yDot) / self.std_yDot
                        train_yDDot = (train_yDDot - self.mean_yDDot) / self.std_yDDot
                    
                    pred_y, pred_yDot, pred_yDDot = self.forward(torch.cat((train_x, train_params), dim=1))
                
                else:
                    train_x, train_y, train_yDot, train_yDDot = trainbatch
                    train_x = train_x.cuda()
                    train_y = train_y.cuda()
                    train_yDot = train_yDot.cuda()
                    train_yDDot = train_yDDot.cuda()
                    
                    with torch.no_grad():
                        train_x = (train_x - self.mean_x) / self.std_x
                        train_y = (train_y - self.mean_y) / self.std_y
                        train_yDot = (train_yDot - self.mean_yDot) / self.std_yDot
                        train_yDDot = (train_yDDot - self.mean_yDDot) / self.std_yDDot
                    
                    pred_y, pred_yDot, pred_yDDot = self.forward(train_x)
                
                # Loss
                loss_y = self.loss_fn(pred_y, train_y)
                loss_yDot = self.loss_fn(pred_yDot, train_yDot)
                loss_yDDot = self.loss_fn(pred_yDDot, train_yDDot)
                
                # Regularizer
                regularizer = 0
                if reg_lambda:
                    for name, param in self.named_parameters():
                        if 'weight' in name:
                            regularizer += torch.sum(torch.square(param))
                    regularizer *= reg_lambda
                
                # Backward
                for param in self.parameters():
                    param.grad = None
                (loss_y + loss_yDot + loss_yDDot + regularizer).backward()
                self.optimizer1.step()
                self.optimizer2.step()
                self.optimizer3.step()
                batch_end = time.time()
                batch_computation_time = batch_end - batch_start
                epoch_computation_time += batch_computation_time
                
                # Print
                if (epoch + 1) % print_every == 0 and int(len(self.train_dataloader) / 3) != 0 and (
                        trainbatch_idx + 1) % int(len(self.train_dataloader) / 3) == 0:
                    print(
                        f"Batch {trainbatch_idx + 1}/{len(self.train_dataloader)} Value MSE: {loss_y:.6f}, {loss_yDot:.6f}, {loss_yDDot:.6f}")
                
                batch_labels = torch.cat((train_y, train_yDot, train_yDDot), dim=1)
                batch_preds = torch.cat((pred_y, pred_yDot, pred_yDDot), dim=1)
                
                tLabel.append(batch_labels.detach().cpu())
                tPrediction.append(batch_preds.detach().cpu())
            
            tLabel = torch.cat(tLabel, dim=0)
            tPrediction = torch.cat(tPrediction, dim=0)
            
            tLoss_y = self.MSE((tLabel - tPrediction)[:, :len(y)])
            tLoss_yDot = self.MSE((tLabel - tPrediction)[:, 1 * len(y):2 * len(y)])
            tLoss_yDDot = self.MSE((tLabel - tPrediction)[:, 2 * len(y):3 * len(y)])
            tLoss_yDDot_ = self.MSE(
                    (tLabel - tPrediction)[:, 2 * len(y):3 * len(y)] * self.std_yDDot.cpu() + self.mean_yDDot.cpu())
            
            # Validation step
            vLabel = []
            vPrediction = []
            self.eval()
            with torch.no_grad():
                for validbatch in self.valid_dataloader:
                    if self.model_info['metamodeling']:
                        valid_x, valid_params, valid_y, valid_yDot, valid_yDDot = validbatch
                        valid_x = valid_x.cuda()
                        valid_params = valid_params.cuda()
                        valid_y = valid_y.cuda()
                        valid_yDot = valid_yDot.cuda()
                        valid_yDDot = valid_yDDot.cuda()
                        
                        valid_x = (valid_x - self.mean_x) / self.std_x
                        valid_params = (valid_params - self.mean_params) / self.std_params
                        valid_y = (valid_y - self.mean_y) / self.std_y
                        valid_yDot = (valid_yDot - self.mean_yDot) / self.std_yDot
                        valid_yDDot = (valid_yDDot - self.mean_yDDot) / self.std_yDDot
                        
                        pred_y, pred_yDot, pred_yDDot = self.forward(torch.cat((valid_x, valid_params), dim=1))
                    
                    else:
                        valid_x, valid_y, valid_yDot, valid_yDDot = validbatch
                        valid_x = valid_x.cuda()
                        valid_y = valid_y.cuda()
                        valid_yDot = valid_yDot.cuda()
                        valid_yDDot = valid_yDDot.cuda()
                        
                        with torch.no_grad():
                            valid_x = (valid_x - self.mean_x) / self.std_x
                            valid_y = (valid_y - self.mean_y) / self.std_y
                            valid_yDot = (valid_yDot - self.mean_yDot) / self.std_yDot
                            valid_yDDot = (valid_yDDot - self.mean_yDDot) / self.std_yDDot
                        
                        pred_y, pred_yDot, pred_yDDot = self.forward(valid_x)
                    
                    batch_labels = torch.cat((valid_y, valid_yDot, valid_yDDot), dim=1)
                    batch_preds = torch.cat((pred_y, pred_yDot, pred_yDDot), dim=1)
                    
                    vLabel.append(batch_labels.detach().cpu())
                    vPrediction.append(batch_preds.detach().cpu())
            self.train()
            
            vLabel = torch.cat(vLabel, dim=0)
            vPrediction = torch.cat(vPrediction, dim=0)
            vLoss_y = self.MSE((vLabel - vPrediction)[:, :len(y)])
            vLoss_yDot = self.MSE((vLabel - vPrediction)[:, len(y):2 * len(y)])
            vLoss_yDDot = self.MSE((vLabel - vPrediction)[:, 2 * len(y):3 * len(y)])
            vLoss_yDDot_ = self.MSE(
                    (vLabel - vPrediction)[:, 2 * len(y):3 * len(y)] * self.std_yDDot.cpu() + self.mean_yDDot.cpu())
            if valid_measure == 'mse':
                vLoss = self.MSE(vLabel - vPrediction)
            elif valid_measure == 'rms':
                vLoss = self.RMS(vLabel, vPrediction)
            try:
                R2value = r2_score(vLabel, vPrediction, multioutput='raw_values')
            except ValueError:
                R2value = torch.zeros(vLabel.shape[1])
                warnings.warn(f"R2 calculation encountered NaN", UserWarning)
            
            tLoss_yDDot_history[epoch] = tLoss_yDDot_.item()
            vLoss_yDDot_history[epoch] = vLoss_yDDot_.item()
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1} (done in {epoch_computation_time:.4f} sec)")
                print(f"Learning rate={self.optimizer1.param_groups[0]['lr']:.3e}")
                print(f"Regularizer = {regularizer:.6f}")
                print(f"Normalized loss1 = Train:{tLoss_y.item():.6f}, Valid:{vLoss_y.item():.6f}")
                print(f"Normalized loss2 = Train:{tLoss_yDot.item():.6f}, Valid:{vLoss_yDot.item():.6f}")
                print(f"Normalized loss3 = Train:{tLoss_yDDot.item():.6f}, Valid:{vLoss_yDDot.item():.6f}")
                print(f"Validation measure({valid_measure.upper()}) = {vLoss.item():.6f}")
                for outidx, r2value in enumerate(R2value):
                    print(f"R2({(y + yDot + yDDot)[outidx]})={r2value:.6f}")
                print()
            
            # Save state dict
            if (epoch + 1) % save_every == 0:
                model_parameter = self.state_dict()
                for name, param in model_parameter.items():
                    model_parameter[name] = param.cpu()
                model_validation_loss[epoch] = vLoss.item()
                model_parameter_group[epoch] = model_parameter
            
            # LR decay
            if (epoch + 1) % lr_halflife == 0:
                old_LR = self.optimizer1.param_groups[0]['lr']
                self.optimizer1.param_groups[0]['lr'] /= 2
                self.optimizer2.param_groups[0]['lr'] /= 2
                self.optimizer3.param_groups[0]['lr'] /= 2
                new_LR = self.optimizer1.param_groups[0]['lr']
                print(f"LR decayed, {old_LR:.3e} -> {new_LR:.3e}")
                print()
        
        # End of training epoch
        TrainingEndTime = time.time()
        Hr, Min, Sec = self.Sec2Time(TrainingEndTime - TrainingStartTime)
        print(f"Training finished in {Hr}hr {Min}min {Sec}sec.")
        self.count_params()
        if save:
            idx_argmin = np.argmin(model_validation_loss)
            print(f"Saving the best model:")
            print(
                f"Validation loss({valid_measure.upper()}) {model_validation_loss[idx_argmin]:.6f} at Epoch {idx_argmin + 1} ")
            
            self.model_info['training_time'] = (TrainingEndTime - TrainingStartTime)
            self.model_info['yddot_training_loss_history'] = tLoss_yDDot_history
            self.model_info['yddot_validation_loss_history'] = vLoss_yDDot_history
            self.model_info['model_state_dict'] = model_parameter_group[idx_argmin]
            self.model_info = {key: value for key, value in sorted(self.model_info.items())}  # arrange
            
            if not os.path.exists('Models'):
                os.mkdir("Models")
            torch.save(self.model_info, f'Models/{filename}')
            # Dictionary txt
            f = open(f"Models/{filename.split('.')[0]}.txt", "w")
            lines = []
            exceptions = ["model_state_dict", ]
            for k in self.model_info.keys():
                if k in exceptions:
                    continue
                else:
                    lines.append(f"{k} : {self.model_info[k]}\n")
            f.writelines(lines)
            f.close()
            
            if model_parameter_group[idx_argmin] == None:
                print(f"WARNING: Saved model parameter is None")
    
    def forward_with_normalization(self, x):
        self.eval()
        x = x.cuda()
        if self.metamodeling:
            x = (x - torch.cat((self.mean_x, self.mean_params))) / torch.cat((self.std_x, self.std_params))
        else:
            x = (x - self.mean_x) / self.std_x
        with torch.no_grad():
            pred_y, pred_yDot, pred_yDDot = self.forward(x)
        pred_y = (pred_y * self.std_y + self.mean_y).cpu()
        pred_yDot = (pred_yDot * self.std_yDot + self.mean_yDot).cpu()
        pred_yDDot = (pred_yDDot * self.std_yDDot + self.mean_yDDot).cpu()
        return pred_y, pred_yDot, pred_yDDot
    
    def forward_without_normalization(self, x):
        self.eval()
        x = x.cuda()
        if self.metamodeling:
            x = (x - torch.cat((self.mean_x, self.mean_params))) / torch.cat((self.std_x, self.std_params))
        else:
            x = (x - self.mean_x) / self.std_x
        with torch.no_grad():
            pred_y, pred_yDot, pred_yDDot = self.forward(x)
        pred_y = (pred_y).cpu()
        pred_yDot = (pred_yDot).cpu()
        pred_yDDot = (pred_yDDot).cpu()
        return pred_y, pred_yDot, pred_yDDot
