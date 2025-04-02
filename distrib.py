import math
import torch


class MixtureGaussian():
    def __init__(self, probs, loc, scale):
        assert len(probs) == len(loc) == len(scale)
        probs, loc, scale = torch.tensor(probs), torch.tensor(loc), torch.tensor(scale)
        self.d_membership = torch.distributions.categorical.Categorical(probs=probs)
        self.distribs = [torch.distributions.normal.Normal(l, s) for l, s in zip(loc, scale)]

    def sample(self, N):
        cls = self.d_membership.sample((N,))
        data = torch.cat([distrib.sample((N, 1)) for distrib in self.distribs], dim=1)
        val = data[torch.arange(N), cls]
        return cls, val


class Ring():
    def __init__(self, r1, r2):
        assert r2 > r1, f'{r1=}, {r2=}'
        self.r1 = r1
        self.d = r2 - r1

    def sample(self, N):
        R = self.r1 + torch.rand(N) * self.d
        angle = 2*math.pi*torch.rand(N)

        X, Y = R * torch.cos(angle), R * torch.sin(angle)

        return torch.cat([X.view(-1, 1), Y.view(-1, 1)], dim=1)


class Cross():
    def __init__(self, l, w):
        self.l = l
        self.w = w

    def sample(self, N):
        p1 = 2*self.l*torch.rand(N, 1) - self.l
        p2 = 2*self.w*torch.rand(N, 1) - self.w

        membership = (torch.rand(N, 1) > .5)
        return torch.where(membership, torch.cat([p1, p2], 1), torch.cat([p2, p1], 1))
