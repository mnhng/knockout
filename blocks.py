import itertools
import math

import torch
import torch.nn as nn
import torch.distributions as td


class KnockoutEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, p, **kwargs):
        super().__init__(num_embeddings+1, embedding_dim, **kwargs)
        self.p = p

    def forward(self, X):
        if self.training:
            X = torch.where(torch.empty_like(X).bernoulli_(p=self.p).bool(), X, torch.nan)

        Z0 = self.num_embeddings-1
        aug_input = torch.where(torch.isnan(X), Z0, X).int()
        return super().forward(aug_input)


class ImputeEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, impute_val, p, enable_at_training=True, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.impute_val = impute_val
        self.p = p
        self.enable_at_training = enable_at_training

    def forward(self, X):
        if self.enable_at_training and self.training:
            X = torch.where(torch.empty_like(X).bernoulli_(p=self.p).bool(), X, torch.nan)

        aug_input = torch.where(torch.isnan(X), self.impute_val, X).int()
        return super().forward(aug_input)


class KnockoutContinuousUnbounded(nn.Module):
    def __init__(self, mean, std, gap, p=None):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean).reshape(1, -1).float())
        self.register_buffer('std', torch.as_tensor(std).reshape(1, -1).float())
        self.register_buffer('gap', torch.tensor(gap).float())
        self.p = 0.5**(1./self.mean.shape[1]) if p is None else p

    def forward(self, X):
        if self.training:
            X = torch.where(torch.empty_like(X).bernoulli_(p=self.p).bool(), X, torch.nan)

        M1, M2, normalized = torch.isnan(X), torch.isinf(X), (X - self.mean)/self.std
        return torch.where(M1, self.gap, torch.where(M2, -self.gap, normalized))


class ImputeContinuous(nn.Module):
    def __init__(self, impute_values, p=None, enable_at_training=True):
        super().__init__()
        self.register_buffer('val', torch.as_tensor(impute_values).reshape(1, -1).float())
        self.p = 0.5**(1./self.val.shape[1]) if p is None else p
        self.enable_at_training = enable_at_training

    def forward(self, X):
        if self.enable_at_training and self.training:
            X = torch.where(torch.empty_like(X).bernoulli_(p=self.p).bool(), X, torch.nan)

        return torch.where(torch.isnan(X), self.val, X)


class ZeroImpute(nn.Module):
    def forward(self, X):
        mask = torch.isnan(X)
        imputed_X = torch.where(mask, 0, X)
        return torch.cat([imputed_X, mask.float()], dim=1)


class skImputeLayer(nn.Module):
    def __init__(self, imputer):
        super().__init__()
        self.imputer = imputer

    def forward(self, X):
        out = self.imputer.transform(X.cpu())
        return torch.tensor(out, dtype=X.dtype, device=X.device)


class MIWAE(nn.Module):
    def __init__(self, d_latent, d_hidden, n_features, offset=0.001):
        super().__init__()
        self.d_latent = d_latent
        self.n_features = n_features
        self.encoder = nn.Sequential(
            nn.Linear(n_features, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 2*d_latent),  # the encoder will output both the mean and the diagonal covariance
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 3*n_features),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*n_features)
        )

        self.sp = nn.Softplus()
        self.offset = offset

    def encode(self, input_):
        loc, scale = torch.chunk(self.encoder(input_), 2, dim=-1)
        return td.Independent(td.Normal(loc=loc, scale=self.sp(scale)), 1)

    def decode(self, zgivenx):
        dec_out = self.decoder(zgivenx.reshape(-1, self.d_latent))

        means, scales, deg_freedom = torch.chunk(dec_out, 3, dim=-1)
        scales = self.sp(scales) + self.offset
        deg_freedom = self.sp(deg_freedom) + 3

        return td.StudentT(loc=means, scale=scales, df=deg_freedom)

    def estimate(self, X, no_samples=100):
        dev = next(self.parameters()).device
        X = X.to(dev)
        mask = ~torch.isnan(X)
        input_ = torch.where(mask, X, 0)

        q_z_given_x = self.encode(input_)
        zgivenx = q_z_given_x.rsample([no_samples])
        q_x_given_z = self.decode(zgivenx)

        all_log_pxgivenz = q_x_given_z.log_prob(input_.repeat(no_samples, 1)).reshape(-1, self.n_features)

        tiledmask = mask.repeat(no_samples, 1)

        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask, 1).reshape(no_samples, len(input_))
        p_z = td.Normal(loc=torch.zeros(self.d_latent, device=dev), scale=torch.ones(self.d_latent, device=dev))

        logpz = td.Independent(p_z, 1).log_prob(zgivenx)
        logq = q_z_given_x.log_prob(zgivenx)

        return logpxobsgivenz + logpz - logq, q_x_given_z

    def NLL(self, X, no_samples=100):
        out, _ = self.estimate(X, no_samples)

        return torch.mean(math.log(no_samples) - torch.logsumexp(out, 0))

    def forward(self, X, no_samples=100):
        out, q_x_given_z = self.estimate(X, no_samples)

        imp_weights = torch.softmax(out, 0)  # these are w_1,....,w_L for all observations in the batch

        xms = td.Independent(q_x_given_z, 1).sample().reshape(no_samples, -1, self.n_features)

        return torch.einsum('ki,kij->ij', imp_weights, xms)

    def fit(self, X, n_epochs, no_samples, lr):
        prob = 0.5**(1./self.n_features)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for ep in range(n_epochs):
            optimizer.zero_grad()
            self.zero_grad()
            X_tmp = torch.where(torch.empty_like(X).bernoulli_(p=prob).bool(), X, torch.nan)
            self.NLL(X_tmp, no_samples=no_samples).backward()
            optimizer.step()
            if (ep + 1) % 500 == 0:
                self.eval()
                with torch.inference_mode():
                    NLL = float(self.NLL(X, no_samples=no_samples))
                print(f'ep={ep+1}: MIWAE {NLL=:.4f}')
            self.train()