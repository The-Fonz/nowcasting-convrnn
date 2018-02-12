import torch


def loss_nats(z, y):
    """
    Calculate loss in nats/frame, as described in VPN paper.
    """
    return - torch.sum(z * torch.log(y) + (1-z) * torch.log(1-y))
