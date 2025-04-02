#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

from blocks import (
    KnockoutEmbedding,
    ImputeEmbedding,
    KnockoutContinuousUnbounded,
    ImputeContinuous,
)
from distrib import MixtureGaussian
from util import empirical_marginals, js_div

EMB_DIM = 10


class DualInput(nn.Module):
    def __init__(self, cat_emb, cont_emb):
        super().__init__()
        self.cat_emb = cat_emb
        self.cont_emb = cont_emb
        self.trunk = nn.Sequential(
                nn.Linear(EMB_DIM+1, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 1)
            )

    def forward(self, X):
        inputs = [self.cat_emb(X[:, 0]), self.cont_emb(X[:, 1:])]

        return self.trunk(torch.cat([m.view(len(X), -1) for m in inputs], dim=1))

    def loss_fn(self, X, Y):
        return nn.functional.binary_cross_entropy_with_logits(self(X).squeeze(), Y)

    def count_errors(self, X, Y):
        self.eval()
        dev = next(self.parameters()).device
        errors = torch.empty(X.shape[1]+1, device=dev)

        indices = torch.arange(X.shape[1], device=dev).view(1, -1)
        for i in range(X.shape[1]):
            with torch.inference_mode():
                pred = self(torch.where(indices == i, X, torch.nan)).squeeze() > 0
            errors[i] = (pred != Y.squeeze()).sum()

        with torch.inference_mode():
            pred = self(X).squeeze() > 0
        errors[-1] = (pred != Y.squeeze()).sum()

        return errors

    def predict_marginal(self, *components):
        self.eval()
        dev = next(self.parameters()).device
        indices = torch.arange(len(components), device=dev)

        marginals = []
        with torch.inference_mode():
            for i, c in enumerate(components):
                logit = self(torch.where(indices == i, c.to(dev), torch.nan))
                marginals.append(torch.sigmoid(logit).cpu())

        return marginals


class SingleInputEnsemble(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.first = nn.Sequential(
                nn.Embedding(num_embeddings, EMB_DIM),
                nn.Linear(EMB_DIM, 100), nn.ReLU(),
                nn.Linear(100, 100), nn.ReLU(),
                nn.Linear(100, 1)
            )
        self.second = nn.Sequential(
                nn.Linear(1, 100), nn.ReLU(),
                nn.Linear(100, 100), nn.ReLU(),
                nn.Linear(100, 1)
            )

    def forward(self, X):
        raise NotImplementedError()

    def loss_fn(self, X, Y):
        l1 = nn.functional.binary_cross_entropy_with_logits(self.first(X[:, 0].int()).squeeze(), Y)
        l2 = nn.functional.binary_cross_entropy_with_logits(self.second(X[:, 1:]).squeeze(), Y)
        return l1 + l2

    def count_errors(self, X, Y):
        self.eval()
        dev = next(self.parameters()).device
        errors = torch.full((X.shape[1]+1,), torch.nan, device=dev)

        with torch.inference_mode():
            errors[0] = ((self.first(X[:, 0].int()).squeeze() > 0) != Y.squeeze()).sum()
            errors[1] = ((self.second(X[:, 1:]).squeeze() > 0) != Y.squeeze()).sum()

        return errors

    def predict_marginal(self, *components):
        self.eval()
        dev = next(self.parameters()).device
        assert len(components) == 2

        marginals = []
        with torch.inference_mode():
            marginals.append(torch.sigmoid(self.first(components[0].int().to(dev))).cpu())
            marginals.append(torch.sigmoid(self.second(components[1].to(dev))).cpu())

        return marginals


def train(model, inp, out, lr, nb_it, print_progress=0):
    dev = next(model.parameters()).device
    X, Y = inp.to(dev), out.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(nb_it):
        opt.zero_grad()
        loss = model.loss_fn(X, Y)
        loss.backward()
        opt.step()
        if print_progress > 0 and (it + 1) % print_progress == 0:
            model.eval()
            with torch.inference_mode():
                nloss = model.loss_fn(X, Y)
            print(f'it={it+1} {loss=:.3f} {nloss=:.3f}')
            model.train()

    model.eval()
    with torch.inference_mode():
        print(f'training loss: {model.loss_fn(X, Y):.4f}')


def evaluate(model, inp, out):
    model.eval()
    dev = next(model.parameters()).device
    Xt, Yt = inp.to(dev), out.to(dev)
    dataset = torch.utils.data.TensorDataset(Xt, Yt)
    batches = torch.utils.data.DataLoader(dataset, batch_size=2000)

    errors = torch.zeros(Xt.shape[1]+1, device=dev)
    for Xb, Yb in batches:
        errors += model.count_errors(Xb, Yb)

    return errors / Xt.shape[0]


def generate_data(N):
    X1 = torch.distributions.categorical.Categorical(probs=torch.tensor([.4, .3, .3])).sample((N,))

    cls2, P2 = MixtureGaussian([.3, .4, .3], [.15, .5, .85], [.07, .07, .07]).sample(N)
    cls1, P1 = MixtureGaussian([.3, .4, .3], [.20, .5, .80], [.05, .05, .05]).sample(N)
    cls0, P0 = MixtureGaussian([.3, .4, .3], [.25, .5, .75], [.03, .03, .03]).sample(N)

    X2 = 9*torch.cat([P0.view(N, 1), P1.view(N, 1), P2.view(N, 1)], dim=1)[torch.arange(N), X1] - 4.5

    X = torch.cat([X1.float().view(-1, 1), X2.view(-1, 1)], dim=1)

    Y = torch.cat([(cls0!=1).view(N, 1), (cls1==1).view(N, 1), (cls2==1).view(N, 1)], dim=1)[torch.arange(N), X1].float()

    return X, Y


def eval_marginals(pred, truth):
    def to_prob(series):
        return torch.cat([series.view(-1, 1), (1-series).view(-1, 1)], dim=1)

    return [js_div(to_prob(A), to_prob(B)) for A, B in zip(pred, truth)]


def visualize(dX, dY, aa1, aa2, something):
    pCOLORS = np.array(['blue', 'green'])
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [1, 4]}, dpi=300)

    axes[1][1].scatter(dX[:, 0], dX[:, 1], c=pCOLORS[dY.int()], marker='.', alpha=.7)

    MARKERS = np.array(['.'])
    HANDLES = [tuple([Line2D([], [], color=c, marker=m, linestyle='') for m in MARKERS]) for c in pCOLORS]
    axes[1][1].legend(HANDLES, ['Y=0', 'Y=1'], labelspacing=.1, handler_map={tuple: HandlerTuple(ndivide=None)})

    LABELS = {'E': 'Empirical marg.', 'M1': "Imputation", 'M2': r'Knockout$^*$', 'MX': 'Knockout', 'SE': 'Fitted marg.'}
    w = .1
    for i, (id_, (a, b)) in enumerate(something.items()):
        axes[0][1].bar(aa1+(i-2)*w, a.squeeze(), align='edge', width=w, label=LABELS.get(id_, id_), alpha=.7)
        axes[1][0].plot(b, aa2, marker='x', alpha=.7)

    axes[0][0].set_visible(False)

    axes[0][1].set_xlim(-.3, 2.3)
    axes[0][1].set_xticks([])
    axes[0][1].set_yticks([0, .5, 1])
    axes[0][1].yaxis.tick_right()
    axes[0][1].set(xlabel=r'$X_1$')

    axes[1][0].invert_xaxis()
    axes[1][0].set_xticks([0, .5, 1])
    axes[1][0].set_ylim(-5, 5)
    axes[1][0].set_yticks([])

    axes[1][1].yaxis.tick_right()
    axes[1][1].set_xlim(-.3, 2.3)
    axes[1][1].set_xticks([0, 1, 2])
    axes[1][1].set_ylim(-5, 5)
    axes[1][1].set(ylabel=r'$X_2$')

    fig.tight_layout()

    axes[0][1].legend(labelspacing=.1, handlelength=1., handletextpad=.4,
                      bbox_to_anchor=(-.39, 1.2), loc='upper left')

    fig.savefig('sim3.pdf')


def run_exp(N0=30000, N1=3000):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    nX, nY = generate_data(N0)
    indices = torch.randperm(len(nX))
    X_tn, Y_tn = nX[indices[:N1]], nY[indices[:N1]]
    X_tt, Y_tt = nX[indices[N1:]], nY[indices[N1:]]

    plot_data = {}

    X1_range = torch.tensor([-.5, .5, 1.5, 2.5])
    X2_range = torch.linspace(-4.5, 4.5, 22)

    aa1 = ((X1_range[:-1] + X1_range[1:])/2).view(-1, 1)
    aa2 = ((X2_range[:-1] + X2_range[1:])/2).view(-1, 1)

    plot_data['E'] = empirical_marginals(nX, nY, X1_range, X2_range)

    mean_arr = nX[:, 1:].mean(dim=0, keepdims=True)
    std_arr = nX[:, 1:].std(dim=0, keepdims=True)
    p = 0.5**(1/2)

    # NOTE: M1, bimodal model, no replacement
    M1 = DualInput(ImputeEmbedding(3, EMB_DIM, impute_val=0, p=p, enable_at_training=False), ImputeContinuous(mean_arr, p=p, enable_at_training=False)).to(dev)
    train(M1, X_tn, Y_tn, lr=3e-3, nb_it=5000)
    plot_data['M1'] = M1.predict_marginal(aa1, aa2)
    print(eval_marginals(plot_data['M1'], plot_data['E']))
    print(evaluate(M1, X_tt, Y_tt))

    # NOTE: M2, bimodal model, replace with mean
    M2 = DualInput(ImputeEmbedding(3, EMB_DIM, impute_val=0, p=p, enable_at_training=True), ImputeContinuous(mean_arr, p=p, enable_at_training=True)).to(dev)
    train(M2, X_tn, Y_tn, lr=3e-3, nb_it=5000, print_progress=1000)
    plot_data['M2'] = M2.predict_marginal(aa1, aa2)
    print(eval_marginals(plot_data['M2'], plot_data['E']))
    print(evaluate(M2, X_tt, Y_tt))

    # NOTE: MX, bimodal model, replace with OOD
    MX = DualInput(KnockoutEmbedding(3, EMB_DIM, p=p), KnockoutContinuousUnbounded(mean_arr, std_arr, gap=10, p=p)).to(dev)
    train(MX, X_tn, Y_tn, lr=3e-3, nb_it=5000, print_progress=1000)
    plot_data['MX'] = MX.predict_marginal(aa1, aa2)
    print(eval_marginals(plot_data['MX'], plot_data['E']))
    print(evaluate(MX, X_tt, Y_tt))

    # NOTE: SE, unimodal models
    SE = SingleInputEnsemble(num_embeddings=3).to(dev)
    train(SE, X_tn, Y_tn, lr=3e-3, nb_it=5000, print_progress=1000)
    plot_data['SE'] = SE.predict_marginal(aa1, aa2)
    print(eval_marginals(plot_data['SE'], plot_data['E']))
    print(evaluate(SE, X_tt, Y_tt))

    visualize(X_tn, Y_tn, aa1.squeeze(), aa2, plot_data)


if __name__ == '__main__':
    torch.random.manual_seed(1)
    run_exp()
