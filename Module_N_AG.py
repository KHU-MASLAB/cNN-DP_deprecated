import torch, time, warnings, os
import numpy as np
from Module_N_C import N_C
from sklearn.metrics import r2_score
from Vars import *

class N_AG(N_C):
    def fit(self,
            epochs,
            lr_halflife,
            filename,
            save: bool = True,
            save_every: int = 1,
            reg_lambda: float = 0,
            initial_lr: float = 1e-3,
            loss_fn: str = 'mse',
            optimizer: str = 'radam',
            print_every: int = 10,
            valid_measure: str = 'mse'):
        ################## model info ##################
        vars = locals()
        if 'self' in vars.keys(): del vars['self']
        if '__class__' in vars.keys(): del vars['__class__']
        self.model_info.update(vars)
        self.model_info = {key: value for key, value in sorted(self.model_info.items())}
        ################## model info ##################
        self.setup_optimizer(initial_lr=initial_lr)
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
                        train_yDot = (train_yDot) * (self.std_x / self.std_y)
                        train_yDDot = (train_yDDot) * (self.std_x ** 2 / self.std_y)
                    
                    train_x.requires_grad_(True)
                    pred_y = self.forward(torch.cat((train_x, train_params), dim=1))
                    pred_yDot = []
                    pred_yDDot = []
                    for k in range(len(y)):
                        pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=train_x,
                                                             grad_outputs=torch.ones_like(pred_y[:, k]),
                                                             create_graph=True, retain_graph=True)[0])
                        pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=train_x,
                                                              grad_outputs=torch.ones_like(pred_yDot[k]),
                                                              create_graph=True, retain_graph=True)[0])
                    pred_yDot = torch.cat(pred_yDot, dim=1)
                    pred_yDDot = torch.cat(pred_yDDot, dim=1)
                    train_x.requires_grad_(False)
                
                else:
                    train_x, train_y, train_yDot, train_yDDot = trainbatch
                    train_x = train_x.cuda()
                    train_y = train_y.cuda()
                    train_yDot = train_yDot.cuda()
                    train_yDDot = train_yDDot.cuda()
                    
                    with torch.no_grad():
                        train_x = (train_x - self.mean_x) / self.std_x
                        train_y = (train_y - self.mean_y) / self.std_y
                        train_yDot = (train_yDot) * (self.std_x / self.std_y)
                        train_yDDot = (train_yDDot) * (self.std_x ** 2 / self.std_y)
                    
                    train_x.requires_grad_(True)
                    pred_y = self.forward(train_x)
                    pred_yDot = []
                    pred_yDDot = []
                    for k in range(len(y)):
                        pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=train_x,
                                                             grad_outputs=torch.ones_like(pred_y[:, k]),
                                                             create_graph=True, retain_graph=True)[0])
                        pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=train_x,
                                                              grad_outputs=torch.ones_like(pred_yDot[k]),
                                                              create_graph=True, retain_graph=True)[0])
                    pred_yDot = torch.cat(pred_yDot, dim=1)
                    pred_yDDot = torch.cat(pred_yDDot, dim=1)
                    train_x.requires_grad_(False)
                
                # Loss
                loss_y = self.loss_fn(pred_y, train_y)
                loss_yDot = self.loss_fn(pred_yDot, train_yDot)
                loss_yDDot = self.loss_fn(pred_yDDot, train_yDDot)
                
                # Regularizer
                regularizer = 0
                if reg_lambda:
                    for name, param in self.named_parameters():
                        if 'weight' in name:
                            regularizer += torch.square(param)
                    regularizer *= reg_lambda
                
                # Backward
                for param in self.parameters():
                    param.grad = None
                (loss_y + loss_yDot + loss_yDDot + regularizer).backward()
                self.optimizer1.step()
                batch_end = time.time()
                batch_computation_time = batch_end - batch_start
                epoch_computation_time += batch_computation_time
                
                # Save
                batch_labels = torch.cat((train_y, train_yDot, train_yDDot), dim=1).detach()
                batch_preds = torch.cat((pred_y, pred_yDot, pred_yDDot), dim=1).detach()
                tLabel.append(batch_labels.cpu())
                tPrediction.append(batch_preds.cpu())
                
                # Print
                if (epoch + 1) % print_every == 0 and int(len(self.train_dataloader) / 3) != 0 and (
                        trainbatch_idx + 1) % int(len(self.train_dataloader) / 3) == 0:
                    print(
                        f"Batch {trainbatch_idx + 1}/{len(self.train_dataloader)} loss: {loss_y:.6f}, {loss_yDot:.6f}, {loss_yDDot:.6f}")
            
            tLabel = torch.cat(tLabel, dim=0)
            tPrediction = torch.cat(tPrediction, dim=0)
            tLoss_y = self.MSE((tLabel - tPrediction)[:, :len(y)])
            tLoss_yDot = self.MSE((tLabel - tPrediction)[:, 1 * len(y):2 * len(y)])
            tLoss_yDDot = self.MSE((tLabel - tPrediction)[:, 2 * len(y):3 * len(y)])
            tLoss_yDDot_ = self.MSE(
                    (tLabel - tPrediction)[:, 2 * len(y):3 * len(y)] * (self.std_y.cpu() / self.std_x.cpu() ** 2))
            
            # Validation step
            vLabel = []
            vPrediction = []
            self.eval()
            for validbatch in self.valid_dataloader:
                if self.model_info['metamodeling']:
                    valid_x, valid_params, valid_y, valid_yDot, valid_yDDot = validbatch
                    valid_x = valid_x.cuda()
                    valid_params = valid_params.cuda()
                    valid_y = valid_y.cuda()
                    valid_yDot = valid_yDot.cuda()
                    valid_yDDot = valid_yDDot.cuda()
                    
                    with torch.no_grad():
                        valid_x = (valid_x - self.mean_x) / self.std_x
                        valid_params = (valid_params - self.mean_params) / self.std_params
                        valid_y = (valid_y - self.mean_y) / self.std_y
                        valid_yDot = (valid_yDot) * (self.std_x / self.std_y)
                        valid_yDDot = (valid_yDDot) * (self.std_x ** 2 / self.std_y)
                    
                    valid_x.requires_grad_(True)
                    pred_y = self.forward(torch.cat((valid_x, valid_params), dim=1))
                    pred_yDot = []
                    pred_yDDot = []
                    for k in range(len(y)):
                        pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=valid_x,
                                                             grad_outputs=torch.ones_like(pred_y[:, k]),
                                                             create_graph=True, retain_graph=True)[0])
                        pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=valid_x,
                                                              grad_outputs=torch.ones_like(pred_yDot[k]),
                                                              create_graph=True, retain_graph=True)[0])
                    pred_yDot = torch.cat(pred_yDot, dim=1)
                    pred_yDDot = torch.cat(pred_yDDot, dim=1)
                    valid_x.requires_grad_(False)
                
                else:
                    valid_x, valid_y, valid_yDot, valid_yDDot = validbatch
                    valid_x = valid_x.cuda()
                    valid_y = valid_y.cuda()
                    valid_yDot = valid_yDot.cuda()
                    valid_yDDot = valid_yDDot.cuda()
                    
                    with torch.no_grad():
                        valid_x = (valid_x - self.mean_x) / self.std_x
                        valid_y = (valid_y - self.mean_y) / self.std_y
                        valid_yDot = (valid_yDot) * (self.std_x / self.std_y)
                        valid_yDDot = (valid_yDDot) * (self.std_x ** 2 / self.std_y)
                    
                    valid_x.requires_grad_(True)
                    pred_y = self.forward(valid_x)
                    pred_yDot = []
                    pred_yDDot = []
                    for k in range(len(y)):
                        pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=valid_x,
                                                             grad_outputs=torch.ones_like(pred_y[:, k]),
                                                             create_graph=True, retain_graph=True)[0])
                        pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=valid_x,
                                                              grad_outputs=torch.ones_like(pred_yDot[k]),
                                                              create_graph=True, retain_graph=True)[0])
                    pred_yDot = torch.cat(pred_yDot, dim=1)
                    pred_yDDot = torch.cat(pred_yDDot, dim=1)
                    valid_x.requires_grad_(False)
                
                # Save
                batch_labels = torch.cat((valid_y, valid_yDot, valid_yDDot), dim=1).detach()
                batch_preds = torch.cat((pred_y, pred_yDot, pred_yDDot), dim=1).detach()
                vLabel.append(batch_labels.cpu())
                vPrediction.append(batch_preds.cpu())
                self.train()
            
            vLabel = torch.cat(vLabel, dim=0)
            vPrediction = torch.cat(vPrediction, dim=0)
            vLoss_y = self.MSE((vLabel - vPrediction)[:, :len(y)])
            vLoss_yDot = self.MSE((vLabel - vPrediction)[:, len(y):2 * len(y)])
            vLoss_yDDot = self.MSE((vLabel - vPrediction)[:, 2 * len(y):3 * len(y)])
            vLoss_yDDot_ = self.MSE(
                    (vLabel - vPrediction)[:, 2 * len(y):3 * len(y)] * (self.std_y.cpu() / self.std_x.cpu() ** 2))
            if valid_measure == 'mse':
                vLoss = self.MSE(vLabel - vPrediction)
            elif valid_measure == 'rms':
                vLoss = self.RMS(vLabel, vPrediction)
            try:
                R2value = r2_score(vLabel, vPrediction, multioutput='raw_values')
            except ValueError:
                R2value = torch.zeros(vLabel.shape[1])
                warnings.warn(f"R2 calculation encountered NaN")
            
            tLoss_yDDot_history[epoch] = tLoss_yDDot_.item()
            vLoss_yDDot_history[epoch] = vLoss_yDDot_.item()
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1} (done in {epoch_computation_time:.4f} sec)")
                print(f"Learning rate={self.optimizer1.param_groups[0]['lr']:.3e}")
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
            t = x[:, 0].view(-1, 1)
            t.requires_grad_(True)
            pred_y = self.forward(torch.cat((t, x[:, 1:]), dim=1))
            pred_yDot = []
            pred_yDDot = []
            for k in range(len(y)):
                pred_yDot.append(
                        torch.autograd.grad(outputs=pred_y[:, k], inputs=t, grad_outputs=torch.ones_like(pred_y[:, k]),
                                            create_graph=True, retain_graph=True)[0])
                pred_yDDot.append(
                        torch.autograd.grad(outputs=pred_yDot[k], inputs=t, grad_outputs=torch.ones_like(pred_yDot[k]),
                                            create_graph=True, retain_graph=True)[0])
            pred_yDot = torch.cat(pred_yDot, dim=1)
            pred_yDDot = torch.cat(pred_yDDot, dim=1)
            t.requires_grad_(False)
        else:
            x = (x - self.mean_x) / self.std_x
            t = x
            t.requires_grad_(True)
            pred_y = self.forward(t)
            pred_yDot = []
            pred_yDDot = []
            for k in range(len(y)):
                pred_yDot.append(
                        torch.autograd.grad(outputs=pred_y[:, k], inputs=t, grad_outputs=torch.ones_like(pred_y[:, k]),
                                            create_graph=True, retain_graph=True)[0])
                pred_yDDot.append(
                        torch.autograd.grad(outputs=pred_yDot[k], inputs=t, grad_outputs=torch.ones_like(pred_yDot[k]),
                                            create_graph=True, retain_graph=True)[0])
            pred_yDot = torch.cat(pred_yDot, dim=1)
            pred_yDDot = torch.cat(pred_yDDot, dim=1)
            t.requires_grad_(False)
        
        pred_y = (pred_y * self.std_y + self.mean_y).detach().cpu()
        pred_yDot = (pred_yDot * (self.std_y / self.std_x)).detach().cpu()
        pred_yDDot = (pred_yDDot * (self.std_y / self.std_x ** 2)).detach().cpu()
        
        return pred_y, pred_yDot, pred_yDDot
    
    
    def forward_without_normalization(self, x):
        self.eval()
        x = x.cuda()
        if self.metamodeling:
            x = (x - torch.cat((self.mean_x, self.mean_params))) / torch.cat((self.std_x, self.std_params))
            t = x[:, 0].view(-1, 1)
            t.requires_grad_(True)
            pred_y = self.forward(torch.cat((t, x[:, 1:]), dim=1))
            pred_yDot = []
            pred_yDDot = []
            for k in range(len(y)):
                pred_yDot.append(
                        torch.autograd.grad(outputs=pred_y[:, k], inputs=t, grad_outputs=torch.ones_like(pred_y[:, k]),
                                            create_graph=True, retain_graph=True)[0])
                pred_yDDot.append(
                        torch.autograd.grad(outputs=pred_yDot[k], inputs=t, grad_outputs=torch.ones_like(pred_yDot[k]),
                                            create_graph=True, retain_graph=True)[0])
            pred_yDot = torch.cat(pred_yDot, dim=1)
            pred_yDDot = torch.cat(pred_yDDot, dim=1)
            t.requires_grad_(False)
        else:
            x = (x - self.mean_x) / self.std_x
            t = x
            t.requires_grad_(True)
            pred_y = self.forward(t)
            pred_yDot = []
            pred_yDDot = []
            for k in range(len(y)):
                pred_yDot.append(
                        torch.autograd.grad(outputs=pred_y[:, k], inputs=t, grad_outputs=torch.ones_like(pred_y[:, k]),
                                            create_graph=True, retain_graph=True)[0])
                pred_yDDot.append(
                        torch.autograd.grad(outputs=pred_yDot[k], inputs=t, grad_outputs=torch.ones_like(pred_yDot[k]),
                                            create_graph=True, retain_graph=True)[0])
            pred_yDot = torch.cat(pred_yDot, dim=1)
            pred_yDDot = torch.cat(pred_yDDot, dim=1)
            t.requires_grad_(False)
        
        pred_y = (pred_y).detach().cpu()
        pred_yDot = ((pred_yDot * (self.std_y / self.std_x) - self.mean_yDot) / self.std_yDot).detach().cpu()
        pred_yDDot = ((pred_yDDot * (self.std_y / self.std_x ** 2) - self.mean_yDDot) / self.std_yDDot).detach().cpu()
        
        return pred_y, pred_yDot, pred_yDDot
