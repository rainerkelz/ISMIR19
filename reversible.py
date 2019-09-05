import torch
from torch import nn
from torch.nn import init


def split(u):
    u1, u2 = torch.chunk(u, 2, dim=1)
    return u1, u2


def merge(v1, v2):
    return torch.cat([v1, v2], dim=1)


def reverse(u):
    return u.flip(dims=(1, ))


def permutation_matrix(n):
    return torch.eye(n)[torch.randperm(n)]


def transform(w):
    return nn.Sequential(
        nn.Linear(w, w, bias=True),
        nn.ReLU(),
        nn.Linear(w, w, bias=True),
        nn.ReLU(),
        nn.Linear(w, w, bias=True)
    )


class InvBlock(nn.Module):
    def __init__(self, width, clamp):
        super().__init__()
        w = width // 2
        self.s1 = transform(w)
        self.t1 = transform(w)
        self.s2 = transform(w)
        self.t2 = transform(w)
        # self.actnorm = ActNorm(width)

        # TODO: find out if this will be saved, when the model parameters are saved?
        # would be problematic, if otherwise ...
        self.register_buffer('permutation', permutation_matrix(width))
        self.clamp = clamp

    # def e(self, s):
    #     # very, *very* important!
    #     return torch.exp(self.clamp * 0.636 * torch.atan(s))
    def e(self, s):
        # # very, *very* important!
        return torch.exp(self.clamp * 0.636 * torch.atan(s))
        # return torch.exp(s)

    def mix(self, m):
        # quite a bit weirder with the random perm ...
        return m.mm(self.permutation)
        # return m

    def unmix(self, m):
        return m.mm(self.permutation.t())
        # return m

    def encode(self, u):
        # u = self.actnorm.encode(u)
        u = self.mix(u)
        u1, u2 = split(u)

        v1 = self.e(self.s2(u2)) * u1 + self.t2(u2)
        v2 = self.e(self.s1(v1)) * u2 + self.t1(v1)

        return merge(v1, v2)

    def decode(self, v):
        v1, v2 = split(v)
        u2 = (v2 - self.t1(v1)) / self.e(self.s1(v1))
        u1 = (v1 - self.t2(u2)) / self.e(self.s2(u2))

        u = merge(u1, u2)
        u = self.unmix(u)
        return u
        # return self.actnorm.decode(u)


class InvBlockChain(nn.Module):
    def __init__(self, width, depth, clamp):
        super().__init__()
        self.chain = []
        for d in range(depth):
            self.chain.append(InvBlock(width, clamp))
        self.chain = nn.ModuleList(self.chain)
        self.initialize_weights()

    def encode(self, u):
        v = u
        for ib in self.chain:
            v = ib.encode(v)
        return v

    def decode(self, v):
        u = v
        for ib in self.chain[::-1]:
            u = ib.decode(u)
        return u

    def initialize_weights(self):
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    init.kaiming_uniform_(module.weight, init.calculate_gain('relu'))
                    if module.bias is not None:
                        init.constant_(module.bias, 0.)


# ActNorm Layer with data-dependant init
class ActNorm(nn.Module):
    def __init__(self, num_features, logscale_factor=1., scale=1.):
        super(ActNorm, self).__init__()
        self.initialized = False
        self.logscale_factor = logscale_factor
        self.scale = scale
        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1)))
        self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features, 1)))

    def encode(self, x):
        input_shape = x.size()
        x = x.view(input_shape[0], input_shape[1], -1)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.hist(x.detach().cpu().numpy().flatten(), histtype='step', label='x')
        if not self.initialized:
            self.initialized = True
            unsqueeze = lambda x: x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = x.size(0) * x.size(-1)
            input_sum = x.sum(dim=0).sum(dim=-1)
            b = input_sum / sum_size * -1.
            vars = ((x - unsqueeze(b)) ** 2).sum(dim=0).sum(dim=1) / sum_size
            vars = unsqueeze(vars)
            logs = torch.log(self.scale / torch.sqrt(vars) + 1e-6) / self.logscale_factor

            self.b.data.copy_(unsqueeze(b).data)
            self.logs.data.copy_(logs.data)

        logs = self.logs * self.logscale_factor
        b = self.b

        output = (x + b) * torch.exp(logs)

        # ax.hist(output.detach().cpu().numpy().flatten(), histtype='step', label='normed')
        # ax.legend().draggable()
        # plt.show()
        return output.view(input_shape)

    def decode(self, x):
        assert self.initialized
        input_shape = x.size()
        x = x.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        output = x * torch.exp(-logs) - b

        return output.view(input_shape)


# a thin convenience wrapper around the actual pytorch model
# that implements the padded, reversible model
class ReversibleModel(nn.Module):
    def __init__(self,
                 device,
                 batch_size,
                 depth,
                 ndim_tot,
                 ndim_x,
                 ndim_y,
                 ndim_z,
                 clamp,
                 zeros_noise_scale,
                 y_noise_scale):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.depth = depth
        self.ndim_tot = ndim_tot
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y
        self.ndim_z = ndim_z
        self.zeros_noise_scale = zeros_noise_scale
        self.y_noise_scale = y_noise_scale

        self.model = InvBlockChain(ndim_tot, depth, clamp=clamp)

        n_total = 0
        for p in self.model.parameters():
            n_total += torch.prod(torch.tensor(p.size()))
        print('n_total', n_total)

        # IMPORTANT: these need to be initialized with zeros
        self.x_padding = torch.zeros(
            self.batch_size,
            self.ndim_tot - self.ndim_x,
            device=self.device
        )
        self.zy_padding = torch.zeros(
            self.batch_size,
            self.ndim_tot - self.ndim_y - self.ndim_z,
            device=self.device
        )

    def train(self, _train=True):
        # print('model in train mode')
        super().train(_train)

    def eval(self):
        # print('model in eval mode')
        super().eval()

        # IMPORTANT: these need to be reset to zero!
        self.x_padding.zero_()
        self.zy_padding.zero_()

    def encode_padding(self, x, padding):
        if x.size(1) != self.ndim_x:
            raise ValueError('wrong dimensions for x')

        x_full = torch.cat([x, padding], dim=1)

        zy_hat_padded = self.model.encode(x_full)

        z_hat = zy_hat_padded[:, :self.ndim_z]
        zy_hat_padding = zy_hat_padded[:, self.ndim_z:-self.ndim_y]
        y_hat = zy_hat_padded[:, -self.ndim_y:]
        return z_hat, zy_hat_padding, y_hat

    def encode(self, x):
        if self.training:
            self.x_padding.normal_(0, self.zeros_noise_scale)

        return self.encode_padding(x, self.x_padding)

    def decode_padding(self, z, padding, y):
        if z.size(1) != self.ndim_z:
            raise ValueError('wrong dimensions for z')
        if y.size(1) != self.ndim_y:
            raise ValueError('wrong dimensions for y')

        x_padded = self.model.decode(torch.cat([z, padding, y], dim=1))
        return x_padded[:, :self.ndim_x], x_padded[:, self.ndim_x:]

    def decode(self, z, y):
        if self.training:
            self.zy_padding.normal_(0, self.zeros_noise_scale)

        return self.decode_padding(z, self.zy_padding, y)
