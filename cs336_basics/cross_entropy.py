import torch

def cross_entropy(logits, target):
    '''
    logit: [N, V], target: [N]
    '''
    # Using log_sum_exp skill
    N = logits.size(0) # token数，可以是被flatten之后的batch_size * seq_len
    M = logits.max(dim=-1, keepdim=True).values # [N, 1]
    lse = M + torch.log(torch.sum(torch.exp(logits - M), dim=-1, keepdim=True))
    log_probs = logits - lse # [N, V]
    loss = -log_probs[torch.arange(N), target] # 这行相当于loss = torch.empty(N); for i in range(N): loss[i] = -log_probs[i, target[i]]
    return loss.mean()