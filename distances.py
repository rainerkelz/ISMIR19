import torch
from torch import nn
from torch import optim
from torch.nn import init
from torch.nn import functional as F


class Distance(object):
    def __init__(self):
        pass

    def is_adversarial(self):
        return False


class Joint(Distance):
    def __init__(self, distances):
        self.distances = distances

    def __call__(self, x, y):
        d = 0
        for distance in self.distances:
            d += distance(x, y)
        return d

    def __str__(self):
        return ','.join([str(distance) for distance in self.distances])


class PenalizeNegative(Distance):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x, y):
        nx = F.relu(-x)
        ny = F.relu(-y)
        return torch.mean(nx ** 2 + ny ** 2) * self.factor


# this one is not a good idea!
# or it needs its factor very carefully adjusted
class MeanLength(Distance):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x, y):
        nx = torch.norm(x, p=2, dim=1)
        ny = torch.norm(y, p=2, dim=1)
        return torch.mean((nx - ny) ** 2) * self.factor


class MMD(Distance):
    def __init__(self, device, scales=[0.2, 0.5, 0.9, 1.3]):
        super().__init__()
        self.device = device
        self.scales = scales

    def __call__(self, x, y):
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * zz

        XX, YY, XY = (torch.zeros(xx.shape).to(self.device),
                      torch.zeros(xx.shape).to(self.device),
                      torch.zeros(xx.shape).to(self.device))

        # these need to be pulled out of a hat
        for a in self.scales:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

        return torch.mean(XX + YY - 2. * XY)

    def __str__(self):
        return 'MMD({})'.format(','.join(self.scales))


class SWD(Distance):
    def __init__(self, embedding_dim, num_projections, p, device):
        super().__init__()
        self.p = p
        self.projections = torch.FloatTensor(embedding_dim, num_projections).to(device)

    def __sample_projection(self):
        # draw from a unit normal
        self.projections.normal_(0, 1)

        # compute euclidean norms for columns
        norms = torch.norm(self.projections, p=2, dim=0, keepdim=True)

        # re-scale each vector to have length 1
        self.projections = self.projections / norms

    def __call__(self, x, y):
        # draw a new projection matrix
        self.__sample_projection()

        # project both samples 'num_projections' times
        pr_x = x.mm(self.projections)
        pr_y = y.mm(self.projections)

        # sort the projection results
        sorted_pr_x = torch.sort(pr_x, dim=0)[0]
        sorted_pr_y = torch.sort(pr_y, dim=0)[0]

        # compute swd distances
        sliced_wd = torch.pow(sorted_pr_x - sorted_pr_y, self.p)

        # return mean distance
        return sliced_wd.mean()

    def __str__(self):
        return 'SWD({}, {}, {})'.format(self.projections.size(0), self.projections.size(1), self.p)


class WassersteinBox(Distance):
    def __init__(self, embedding_dim, box_constraint, n_critic_updates, device):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        self.critic.to(device)
        self.n_critic_updates = n_critic_updates
        self.optimizer = optim.RMSprop(self.critic.parameters(), lr=1e-6)
        self.box_constraint = box_constraint
        self.initialize_weights()

    def initialize_weights(self):
        with torch.no_grad():
            for module in self.critic.modules():
                if isinstance(module, nn.Linear):
                    print('initializing Adversarial {}'.format(module))
                    init.kaiming_uniform_(module.weight, init.calculate_gain('relu'))
                    if module.bias is not None:
                        init.constant_(module.bias, 0.)

    def enforce_box_constraints(self):
        with torch.no_grad():
            for p in self.critic.parameters():
                p.data = torch.clamp(p.data, -self.box_constraint, self.box_constraint)

    def train(self, sample_fake_x, sample_real_x):
        smoothed_loss = 1.
        for n in range(self.n_critic_updates):
            self.enforce_box_constraints()
            fake_x = next(sample_fake_x)
            real_x = next(sample_real_x)

            self.optimizer.zero_grad()
            d_fake = self.critic(fake_x)
            d_real = self.critic(real_x)
            loss = d_fake.mean() - d_real.mean()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            smoothed_loss = smoothed_loss * 0.9 + loss.detach().cpu().item() * 0.1

        return smoothed_loss

    def distance(self, fake_x, _):
        return -torch.mean(self.critic(fake_x))

    def __call__(self, fake_x, _):
        return self.distance(fake_x, _)

    def is_adversarial(self):
        return True
