#!/usr/bin/env python
import argparse
import itertools
import pathlib
import pickle
import math

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as IterImputer

import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as td
from blocks import *
from util import CondGaussian


def get_model(first_layer, in_dim):
    return nn.Sequential(
            first_layer,
            nn.Linear(in_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1))


def train_test_split(data):
    return data[:, :-1], data[:, [-1]]


def gen_data(out_dir, N0=30000, N1=3000, d=10):
    if out_dir.exists():
        return

    out_dir.mkdir(parents=True)

    mu = torch.rand(size=(d,), dtype=torch.float)
    A = torch.rand(size=(d, d), dtype=torch.float)
    cov = A@A.T

    print(f'{mu=}, {cov=}')

    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=cov)
    cdist = CondGaussian(mu, cov)

    X_full, train_Y = train_test_split(dist.sample((N1,)))
    torch.save(X_full, out_dir/'train_X_full.pt')

    X_MCAR = torch.where(torch.empty_like(X_full).bernoulli_(.1).bool(), torch.nan, X_full)
    torch.save(X_MCAR, out_dir/'train_X_MCAR.pt')

    X_MNAR = torch.where(X_full > torch.quantile(X_full, .9, dim=0, keepdims=True), torch.inf, X_full)
    torch.save(X_MNAR, out_dir/'train_X_MNAR.pt')

    torch.save(train_Y, out_dir/'train_Y.pt')

    torch.save(dist.sample((N0-N1,)), out_dir/'d_test.pt')
    torch.save(mu, out_dir/'d_mu.pt')
    torch.save(cov, out_dir/'d_cov.pt')
    with open(out_dir/'cdist.bin', 'wb') as fh:
        pickle.dump(cdist, fh)


def run_exp(args):
    if args.data == 'full':
        train_X = torch.load(args.out/'train_X_full.pt', weights_only=True)
    elif args.data == 'MCAR':
        train_X = torch.load(args.out/'train_X_MCAR.pt', weights_only=True)
    elif args.data == 'MNAR':
        train_X = torch.load(args.out/'train_X_MNAR.pt', weights_only=True)
        if args.model != 'Knockout':
            train_X = torch.where(torch.isinf(train_X), torch.nan, train_X)

    train_Y = torch.load(args.out/'train_Y.pt', weights_only=True)

    d_test = torch.load(args.out/'d_test.pt', weights_only=True)
    mu = torch.load(args.out/'d_mu.pt', weights_only=True)
    cov = torch.load(args.out/'d_cov.pt', weights_only=True)
    d = d_test.shape[1]
    with open(args.out/'cdist.bin', 'rb') as fh:
        cdist = pickle.load(fh)

    if args.model == 'CB':
        model = get_model(ImputeContinuous(mu[:-1], enable_at_training=False), d-1)

    elif args.model == 'missForest':
        imputer = IterImputer(RandomForestRegressor(), random_state=0)
        imputer.fit(train_X)
        model = get_model(skImputeLayer(imputer), d-1)

    elif args.model == 'MICE':
        imputer = IterImputer(BayesianRidge(), max_iter=100, sample_posterior=False, random_state=0)
        imputer.fit(train_X)
        model = get_model(skImputeLayer(imputer), d-1)

    elif args.model == 'Dropout':
        model = get_model(ImputeContinuous(torch.zeros(d-1), enable_at_training=True), d-1)

    elif args.model == 'Knockout':
        model = get_model(KnockoutContinuousUnbounded(mu[:-1], torch.diag(cov)[:-1], gap=10), d-1)

    elif args.model == 'ZeroImpute':
        model = get_model(ZeroImpute(), 2*(d-1))

    elif args.model in {'MIWAE', 'supMIWAE'}:
        imputer = MIWAE(3, 128, d-1).cuda()
        imputer.fit(train_X, n_epochs=args.nb_it, no_samples=20, lr=args.lr)
        imputer.requires_grad_(False)
        model = get_model(imputer, d-1).cuda()

    if args.model == 'supMIWAE':
        train_supMIWAE(model, train_X, train_Y, lr=args.lr, nb_it=args.nb_it, print_progress=1000)
    else:
        train(model, train_X, train_Y, lr=args.lr, nb_it=args.nb_it, print_progress=1000)

    tag = {'method': args.model, 'rep': args.seed}

    return pd.DataFrame([tag | row for row in evaluate(model, cdist, d_test)])


def train_supMIWAE(model, X, Y, lr, nb_it, print_progress=0):
    dev = next(model.parameters()).device
    X, Y = X.to(dev), Y.to(dev).permute(1, 0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    vae, classifier = model[0], model[1:]

    for it in range(nb_it):
        opt.zero_grad()

        iwae_log_w, q_x_given_z = vae.estimate(X)
        no_samples = len(iwae_log_w)
        xms = td.Independent(q_x_given_z, 1).sample()
        lpyx = -(classifier(xms).reshape(no_samples, -1) - Y)**2

        loss = torch.mean(math.log(no_samples) - torch.logsumexp(lpyx + iwae_log_w, 0))

        loss.backward()
        opt.step()
        if print_progress > 0 and (it + 1) % print_progress == 0:
            print(f'it={it+1} {loss=:.3f}')


def train(model, X, Y, lr, nb_it, print_progress=0):
    dev = next(model.parameters()).device
    X, Y = X.to(dev), Y.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(nb_it):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X), Y)
        loss.backward()
        opt.step()
        if print_progress > 0 and (it + 1) % print_progress == 0:
            model.eval()
            with torch.no_grad():
                nloss = nn.functional.mse_loss(model(X), Y)
            print(f'it={it+1} {loss=:.3f} {nloss=:.3f}')
            model.train()

    model.eval()
    with torch.no_grad():
        loss = nn.functional.mse_loss(model(X), Y)
    print(f'training loss: {loss:.4f}')


def exclude(array, indices):
    return [a for i, a in enumerate(array) if i not in indices]


def evaluate(model, cond_distrib, data):
    dev = next(model.parameters()).device
    model.eval()
    in_idx, out_idx = list(range(data.shape[1]-1)), [data.shape[1]-1]

    Xt, Yt = data[:, in_idx].to(dev), data[:, out_idx].to(dev)
    ret = []
    avg_error, bayes_opt  = {}, {}
    loss_fn = nn.functional.mse_loss
    tag1, tag2 = {'type': 'error'}, {'type': 'bayes'}

    with torch.no_grad():
        Yhat = model(Xt)
        ret.append(tag1 | {'ko': 0} | {'MSE': float(loss_fn(Yhat, Yt))})

        true_cond = cond_distrib.estimate(in_idx, out_idx, data).to(dev)
        ret.append(tag2 | {'ko': 0} | {'MSE': float(loss_fn(Yhat, true_cond))})

    tmp = list(range(Xt.shape[1]))
    for cnt in [1, 2, 3]:
        for i in itertools.combinations(tmp, cnt):
            with torch.no_grad():
                idx_array = torch.zeros((1, Xt.shape[1]), dtype=torch.bool, device=dev)
                idx_array[0, i] = True
                pred_marg = model(torch.where(idx_array, torch.nan, Xt))
                ret.append(tag1 | {'ko': cnt} | {'MSE': float(loss_fn(pred_marg, Yt))})

                true_marg = cond_distrib.estimate(exclude(in_idx, i), out_idx, data).to(dev)
                ret.append(tag2 | {'ko': cnt} | {'MSE': float(loss_fn(pred_marg, true_marg))})

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--nb_it', type=int, default=5000)
    parser.add_argument('--out', '-o', type=pathlib.Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', '-m', choices={'CB', 'missForest', 'MICE', 'Dropout', 'Knockout', 'ZeroImpute', 'MIWAE', 'supMIWAE'}, required=True)
    parser.add_argument('--data', '-d', choices={'full', 'MCAR', 'MNAR'}, required=True)

    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    gen_data(args.out)

    torch.random.manual_seed(args.seed)
    run_exp(args).to_csv(args.out/f'stats_{args.model}.csv', index=False)
