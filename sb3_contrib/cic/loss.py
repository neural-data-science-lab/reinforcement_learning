import math

import torch


def compute_apt_reward(source, target, args):

    b1, b2 = source.size(0), target.size(0)
    # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
    sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)
    reward, _ = sim_matrix.topk(args.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

    if not args.knn_avg:  # only keep k-th nearest neighbor
        reward = reward[:, -1]
        reward = reward.reshape(-1, 1)  # (b1, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward).to(device))  # (b1, )
    else:  # average over all k nearest neighbors
        reward = reward.reshape(-1, 1)  # (b1 * k, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward).to(device))
        reward = reward.reshape((b1, args.knn_k))  # (b1, k)
        reward = reward.mean(dim=1)  # (b1,)
    reward = torch.log(reward + 1.0)
    return reward


def compute_cpc_loss(self, obs, next_obs, skill):
    temperature = self.temp
    eps = 1e-6
    query, key = self.cic.forward(obs, next_obs, skill)
    query = torch.nn.functional.normalize(query, dim=1)
    key = torch.nn.functional.normalize(key, dim=1)
    cov = torch.mm(query, key.T)  # (b,b)
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)  # (b,)
    row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    pos = torch.exp(torch.sum(query * key, dim=-1) / temperature)  # (b,)
    loss = -torch.log(pos / (neg + eps))  # (b,)
    return loss, cov / temperature
