import torch
import torch.nn.functional as F
from torch import nn

def stand_info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='paired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, labels, negative_keys=None):
        return info_nce(query, positive_key, labels, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, labels, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='paired'):
    
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    
    positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True) # [32, 128] * [32, 128] = [32, 1] 

    if negative_mode == 'unpaired':
        # Cosine between all query-negative combinations
        negative_logits = query @ transpose(negative_keys)

    elif negative_mode == 'paired':

        query = query.unsqueeze(1) # [32, 1, 128]
        logits = (query @ transpose(negative_keys)).squeeze(1)  # [32, 1, 128] * [32, 4, 128] = [32, 1, 4] -> [32, 4]


    logits = torch.exp(logits / temperature) # [32, 4]
    logits = logits.sum(dim=0, keepdim=True).squeeze(0) # [4]
    
    positive_logit = torch.exp(positive_logit / temperature).squeeze(1) # [32]
    positive_logit = torch.cat([positive_logit[labels==i].sum(dim=0, keepdim=True) for i in range(len(logits))])

    mask = positive_logit != 0
    positive_logit = positive_logit[mask]
    logits = logits[mask]

    return - torch.log(positive_logit / logits).mean()


class FocalLoss(nn.Module):
    
    def __init__(self, alpha=None, gamma=2, num_classes = 3, size_average=True):

        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1 
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) 

        self.gamma = gamma
        
        
    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        xx = F.softmax(preds, dim=1)
        xx = torch.clamp(xx, min=1e-4, max=1.0)
        preds_logsoft = torch.log(xx) 
        

        preds_softmax = torch.exp(preds_logsoft) # [0, 1]
        
        
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        
        return loss
    
class Sup_InfoNCE(nn.Module):
    def __init__(self, temperature=0.1):
        super(Sup_InfoNCE, self).__init__()
        self.temperature = temperature
        
    def forward(self, query, positive_key, negative_keys=None):
        return contrastive_loss(query, positive_key, negative_keys,
                        temperature=self.temperature)


def contrastive_loss(query, positive_key, negative_keys, temperature=0.1):

    num_samples = query.size(0)

    max_neg = max([neg.size(0) for neg in negative_keys])

    negative_keys_padded = torch.zeros(num_samples, max_neg, query.size(1), device=query.device)  # [N, max_neg, D]
    mask = torch.zeros(num_samples, max_neg, dtype=torch.bool, device=query.device)  # [N, max_neg]

    for i, neg in enumerate(negative_keys):
        negative_keys_padded[i, :neg.size(0)] = neg
        mask[i, :neg.size(0)] = 1  

    positive_logits = (query * positive_key).sum(dim=-1, keepdim=True)  # [N, 1]

    negative_logits = torch.einsum('nd,nkd->nk', query, negative_keys_padded)  # [N, max_neg]

    negative_logits = negative_logits.masked_fill(~mask, float('-inf'))  

    logits = torch.cat([positive_logits, negative_logits], dim=1)  # [N, 1 + max_neg]

    labels = torch.zeros(num_samples, dtype=torch.long, device=query.device)  # [N]

    loss = F.cross_entropy(logits / temperature, labels, reduction='mean')

    return loss


    
        
