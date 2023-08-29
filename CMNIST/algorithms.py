# Code adopted from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py

import torch
import torch.autograd as autograd
from lib.misc import get_grad_norm, Nonparametric

ALGORITHMS = [
    'ERM',
    'EQRM',
    'IRM',
    'GroupDRO',
    'VREx',
    'IGA',
    'SD',
    'RDM',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    globals_lower = {k.lower():v for k, v in globals().items()}
    if algorithm_name not in globals_lower:
        raise NotImplementedError(f"Algorithm not found: {algorithm_name}")
    return globals_lower[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, network, hparams, loss_fn):
        super(Algorithm, self).__init__()
        self.network = network
        self.hparams = hparams
        self.loss_fn = loss_fn
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, network, loss_fn, hparams):
        super(ERM, self).__init__(network, loss_fn, hparams)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer("erm_grad_norm", torch.tensor([-1.]))

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = self.loss_fn(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        if self.erm_grad_norm < 0:              # store initial grad norm for normalizing regularization losses
            self.erm_grad_norm *= -get_grad_norm(self.network)
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class RDM(ERM):
    """Our proposed algorithm"""
    """Domain Generalization via Risk Distribution Matching (we used to be named it DGPM2)"""
    def __init__(self, network, hparams, loss_fn):
        super(RDM, self).__init__(network, hparams, loss_fn)
        
        self.kernel_type='not_gaussian'
        
    def my_cdist(self, x1, x2): #(bs)
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True) 

        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff
    
    @staticmethod
    def _kl_penalty(p_mean, q_mean, p_var, q_var):
        return 0.5 * torch.log(q_var/p_var)+ ((p_var)+(p_mean-q_mean)**2)/(2*q_var) - 0.5
    
    def _js_penalty(self, p_mean, q_mean, p_var, q_var):
        m_mean = (p_mean + q_mean) / 2
        m_var = (p_var + q_var) / 4
        
        return self._kl_penalty(p_mean, m_mean, p_var, m_var) + self._kl_penalty(q_mean, m_mean, q_var, m_var)
    
    def update(self, minibatches, unlabeled=None):
        # ERM pretraining
        if self.update_count < self.hparams["erm_pretrain_iters"]:
            return super(RDM, self).update(minibatches, unlabeled)
        
        # Reset Adam as it doesn't like a sharp jump in gradients when changing objectives
        if self.update_count == self.hparams['erm_pretrain_iters']:
            lr_ = self.hparams["lr"]
            if self.hparams['erm_pretrain_iters'] > 0:
                lr_ /= self.hparams["lr_factor_reduction"]
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=lr_,
                weight_decay=self.hparams['weight_decay'])
    
        # DGPM objective
        
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.predict(all_x)
        losses = torch.zeros(len(minibatches)).cuda()
        all_logits_idx = 0
        all_confs_envs = None
        
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            losses[i] = self.loss_fn(logits, y)
            
            nll = self.loss_fn(logits, y, reduction = "none").unsqueeze(0)
            if all_confs_envs is None:
                all_confs_envs = nll
            else:
                all_confs_envs = torch.cat([all_confs_envs, nll], dim = 0)
            
        erm_loss = losses.mean()
        
        ## squeeze the risks
        all_confs_envs = torch.squeeze(all_confs_envs)
        
        ## find the worst domain
        sum_conf_cls = torch.sum(all_confs_envs, dim=1)
        worst_env_idx = torch.argmax(sum_conf_cls)
        all_confs_worst_env = all_confs_envs[worst_env_idx]
        
        ## flatten the risk
        all_confs_worst_env_flat = torch.flatten(all_confs_worst_env)
        all_confs_all_envs_flat = torch.flatten(all_confs_envs)
    
        matching_penalty = self.mmd(all_confs_worst_env_flat.unsqueeze(1), all_confs_all_envs_flat.unsqueeze(1))
        
        ## variance penalty
        variance_penalty = torch.var(all_confs_worst_env_flat)
        variance_penalty += torch.var(all_confs_all_envs_flat) 
        
        # variance_penalty default is set to 0
        
        total_loss = erm_loss + self.hparams['penalty_weight'] * matching_penalty + self.hparams['variance_weight'] * variance_penalty
        
        if (self.hparams['penalty_weight'] > 1.0):
            total_loss /= self.hparams['penalty_weight']  # Rescale the entire loss to keep gradients in a reasonable range

        # Step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
 
        return {'loss': total_loss.item(), 'erm_loss': erm_loss.item(), 'matching_penalty': matching_penalty.item(), 
                'variance_penalty': variance_penalty.item()}     

class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, network, hparams, loss_fn):
        super(IRM, self).__init__(network, hparams, loss_fn)

    def irm_penalty(self, logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss = self.loss_fn(logits * scale, y)
        grad0 = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad0 ** 2)

    def update(self, minibatches, unlabeled=None):
        # ERM pretraining
        if self.update_count < self.hparams["erm_pretrain_iters"]:
            return super(IRM, self).update(minibatches, unlabeled)

        # Reset Adam as it doesn't like a sharp jump in gradients when changing objectives
        if self.update_count == self.hparams['erm_pretrain_iters']:
            lr_ = self.hparams["lr"]
            if self.hparams['erm_pretrain_iters'] > 0:
                lr_ /= self.hparams["lr_factor_reduction"]
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=lr_,
                weight_decay=self.hparams['weight_decay'])

        # IRM objective
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        erm_loss = 0.           # only ERM here if the envs contain the same number of samples
        penalty = 0.
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            erm_loss += self.loss_fn(logits, y)
            penalty += self.irm_penalty(logits, y)
        erm_loss /= len(minibatches)
        penalty /= len(minibatches)

        loss = erm_loss + (self.hparams['penalty_weight'] * penalty)
        if self.hparams['penalty_weight'] > 1.0:
            loss /= self.hparams['penalty_weight']  # Rescale the entire loss to keep gradients in a reasonable range

        # Step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'erm_loss': erm_loss.item(), 'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, network, hparams, loss_fn):
        super(VREx, self).__init__(network, hparams, loss_fn)

    def update(self, minibatches, unlabeled=None):
        # ERM pretraining
        if self.update_count < self.hparams["erm_pretrain_iters"]:
            return super(VREx, self).update(minibatches, unlabeled)

        # Reset Adam as it doesn't like a sharp jump in gradients when changing objectives
        if self.update_count == self.hparams['erm_pretrain_iters']:
            lr_ = self.hparams["lr"]
            if self.hparams['erm_pretrain_iters'] > 0:
                lr_ /= self.hparams["lr_factor_reduction"]
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=lr_,
                weight_decay=self.hparams['weight_decay'])

        # VREx objective
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        losses = torch.zeros(len(minibatches))
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            losses[i] = self.loss_fn(logits, y)

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + self.hparams["penalty_weight"] * penalty
        if self.hparams["penalty_weight"] > 1.0:
            loss /= self.hparams["penalty_weight"]  # Rescale the entire loss to keep gradients in a reasonable range

        # Step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'erm_loss': mean.item(), 'penalty': penalty.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, network, hparams, loss_fn):
        super(GroupDRO, self).__init__(network, hparams, loss_fn)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        # ERM pretraining
        if self.update_count < self.hparams["erm_pretrain_iters"]:
            return super(GroupDRO, self).update(minibatches, unlabeled)

        # Reset Adam as it doesn't like a sharp jump in gradients when changing objectives
        if self.update_count == self.hparams['erm_pretrain_iters']:
            lr_ = self.hparams["lr"]
            if self.hparams['erm_pretrain_iters'] > 0:
                lr_ /= self.hparams["lr_factor_reduction"]
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=lr_,
                weight_decay=self.hparams['weight_decay'])

        # GroupDRO objective
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)
        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = self.loss_fn(self.network(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()
        loss = torch.dot(losses, self.q)

        # Step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item()}


class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, network, hparams, loss_fn):
        super(IGA, self).__init__(network, hparams, loss_fn)

    def update(self, minibatches, unlabeled=None):
        # ERM pretraining
        if self.update_count < self.hparams["erm_pretrain_iters"]:
            return super(IGA, self).update(minibatches, unlabeled)

        # Reset Adam as it doesn't like a sharp jump in gradients when changing objectives
        if self.update_count == self.hparams['erm_pretrain_iters']:
            lr_ = self.hparams["lr"]
            if self.hparams['erm_pretrain_iters'] > 0:
                lr_ /= self.hparams["lr_factor_reduction"]
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=lr_,
                weight_decay=self.hparams['weight_decay'])

        # IGA objective
        total_loss = 0
        grads = []
        for i, (x, y) in enumerate(minibatches):
            env_loss = self.loss_fn(self.network(x), y)
            total_loss += env_loss

            env_grad = autograd.grad(env_loss, self.network.parameters(), create_graph=True)

            grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)
        mean_grad = autograd.grad(mean_loss, self.network.parameters(), retain_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.hparams['penalty_weight'] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': mean_loss.item(), 'penalty_weight': penalty_value.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, network, hparams, loss_fn):
        super(SD, self).__init__(network, hparams, loss_fn)

    def update(self, minibatches, unlabeled=None):
        # ERM pretraining
        if self.update_count < self.hparams["erm_pretrain_iters"]:
            return super(SD, self).update(minibatches, unlabeled)

        # Reset Adam as it doesn't like a sharp jump in gradients when changing objectives
        if self.update_count == self.hparams['erm_pretrain_iters']:
            lr_ = self.hparams["lr"]
            if self.hparams['erm_pretrain_iters'] > 0:
                lr_ /= self.hparams["lr_factor_reduction"]
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=lr_,
                weight_decay=self.hparams['weight_decay'])

        # SD objective
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_p = self.network(all_x)

        loss = self.loss_fn(all_p, all_y)
        penalty = (all_p ** 2).mean()
        objective = loss + self.hparams['penalty_weight'] * penalty
        if self.hparams['penalty_weight'] > 1.0:
            objective /= self.hparams['penalty_weight']  # Rescale the loss to keep gradients in a reasonable range

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'penalty': penalty.item()}


class EQRM(ERM):
    """
    Empirical quantile risk minimization.
    """
    def __init__(self, network, hparams, loss_fn, dist=Nonparametric()):
        super().__init__(network, hparams, loss_fn)
        self.register_buffer('alpha', torch.tensor(self.hparams["alpha"], dtype=torch.float64))
        self.dist = dist
        self.grad_ratio = None

    def update(self, minibatches, unlabeled=None):
        # ERM pretraining
        if self.update_count < self.hparams["erm_pretrain_iters"]:
            return super().update(minibatches, unlabeled)

        # Reset Adam as it doesn't like a sharp jump in gradients when changing objectives
        if self.update_count == self.hparams['erm_pretrain_iters']:
            lr_ = self.hparams["lr"]
            if self.hparams['erm_pretrain_iters'] > 0:
                lr_ /= self.hparams["lr_factor_reduction"]
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=lr_,
                weight_decay=self.hparams['weight_decay'])

        # QRM objective
        env_risks = torch.cat([self.loss_fn(self.network(x), y).reshape(1) for x, y in minibatches])
        self.dist.estimate_parameters(env_risks)
        loss = self.dist.icdf(self.alpha)

        # Rescale gradients if using erm init/pretraining
        if self.hparams['erm_pretrain_iters'] > 0:
            if self.grad_ratio is None:
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.grad_ratio = get_grad_norm(self.network) / self.erm_grad_norm
            loss = loss / self.grad_ratio

        # Step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item()}

