import torch
from torch import optim, nn, autograd

class IRM_TV_L1_multi_class:
    
    def __init__(self, flags):
        self.flags = flags


    def __call__(self, batch_data, step,  mlp=None, scale=None, mean_nll=None, **kwargs):
        train_x, train_y, train_z, train_g, train_c, train_invnoise = batch_data
        normed_z = (train_z.float() - train_z.float().mean())/train_z.float().std()
        train_logits = scale * mlp(train_x)
        if self.flags.dataset == "house_price":
            loss_fun = nn.MSELoss(reduction='none')
            train_nll = loss_fun(train_logits, train_y)
        elif self.flags.dataset == "landcover": # Not full batch
            train_nll = mean_nll(train_logits, train_y, reduction='none')
        else:
            train_nll = nn.functional.binary_cross_entropy_with_logits(train_logits, train_y, reduction="none")

        ground_envs = train_g
        train_penalty = 0
        if self.flags.dataset == "landcover":   
            train_nll = torch.unsqueeze(train_nll, 1)
            multi_loss = train_nll * ground_envs
            for i in range(multi_loss.shape[1]):
                    grad1 = autograd.grad(
                            multi_loss[:,i][0::2].mean(), 
                            [scale],
                            create_graph=True)[0]
                    grad2 = autograd.grad(
                            multi_loss[:,i][1::2].mean(),
                            [scale],
                            create_graph=True)[0]        
                    train_penalty += (grad1 * grad2).mean().abs().sqrt()
            train_penalty = train_penalty ** 2
        else:
            multi_loss = (train_nll * ground_envs).mean(axis=0) 
            for i in range(multi_loss.shape[0]):
                grad = autograd.grad(
                    multi_loss[i],
                    [scale],
                    create_graph=True)[0]
                train_penalty += grad.abs()
            train_penalty = train_penalty ** 2
    
        train_nll = train_nll.mean()
        return train_nll, train_penalty
    