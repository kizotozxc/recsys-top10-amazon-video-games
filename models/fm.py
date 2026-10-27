"""Second-order factorization machine; PyTorch only supplies tensors and autodiff."""
import numpy as np
import torch
from torch import nn


class FMNetwork(nn.Module):
    def __init__(self, n_features, factors):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.latent = nn.Embedding(n_features, factors)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0., std=0.01)

    def forward(self, fields):
        embeddings = self.latent(fields)
        interaction = 0.5 * (embeddings.sum(1).square() - embeddings.square().sum(1)).sum(1)
        return self.bias + self.linear(fields).sum((1, 2)) + interaction

    def side_terms(self, fields):
        embeddings = self.latent(fields)
        sums = embeddings.sum(1)
        within = 0.5*(sums.square() - embeddings.square().sum(1)).sum(1)
        return sums, self.linear(fields).sum((1, 2)) + within


class FactorizationMachine:
    def __init__(self, factors=32, learning_rate=0.01, regularization=1e-6,
                 epochs=8, batch_size=4096, seed=42, use_metadata=True):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.use_metadata = use_metadata

    def fit(self, events, shape, cutoff, features=None, examples=None):
        from recsys.features import CausalFeatures, make_fm_examples
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        rng = np.random.default_rng(self.seed)
        self.features = features or CausalFeatures(events, shape)
        users, items, times, labels = examples if examples is not None else make_fm_examples(events, shape, self.seed)
        fields = torch.from_numpy(self.features.encode(users, items, times, self.use_metadata))
        labels = torch.from_numpy(labels)
        self.network = FMNetwork(self.features.n_features, self.factors)
        # Adam is a generic optimizer; the FM equation, loss, sampling and scoring are ours.
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.loss_history = []
        self.network.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(fields))
            total = 0.
            for start in range(0, len(order), self.batch_size):
                ix = order[start:start+self.batch_size]
                batch = fields[ix]
                logits = self.network(batch)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, labels[ix])
                # L2 over participating field parameters (not all unused embeddings).
                penalty = self.network.latent(batch).square().sum(2).mean()
                penalty += self.network.linear(batch).square().sum(2).mean()
                loss = bce + self.regularization*penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach())*len(ix)
            self.loss_history.append(total/len(fields))
        self.network.eval()
        # Exact FM decomposition: user-side + item-side + dot(sum(v_user), sum(v_item)).
        # No candidate pruning or approximate retrieval at inference.
        with torch.no_grad():
            uf, itf = self.features.sides_at(cutoff, self.use_metadata)
            ul, ub = self.network.side_terms(torch.from_numpy(uf))
            il, ib = self.network.side_terms(torch.from_numpy(itf))
            self.user_latents, self.user_biases = ul.numpy(), ub.numpy()
            self.item_latents, self.item_biases = il.numpy(), ib.numpy()
            self.global_bias = float(self.network.bias)
        return self

    def score_users(self, users):
        return (self.user_latents[users] @ self.item_latents.T + self.user_biases[users, None]
                + self.item_biases[None, :] + self.global_bias)
