import torch
import torch.nn as nn

class FM_FTRL_Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=8, k=4):
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.V = nn.Parameter(torch.randn(input_dim, k) * 0.01)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        linear_part = self.linear(x)
        fm_interactions = 0.5 * torch.sum(
            (x @ self.V) ** 2 - (x ** 2) @ (self.V ** 2),
            dim=1, keepdim=True
        )
        fm_interactions_expanded = fm_interactions.expand(-1, self.embedding_dim)
        return linear_part + fm_interactions_expanded

class FM_FTRL_WithClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim=8, k=4):
        super().__init__()
        self.encoder = FM_FTRL_Encoder(input_dim, embedding_dim, k)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return torch.sigmoid(logits).squeeze(1), embedding

class EWCWrapper:
    def __init__(self, model, lambda_ewc=1000.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._fisher = {}

    def consolidate(self, dataloader, device):
        self._means = {n: p.clone().detach() for n, p in self.params.items()}
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}

        self.model.eval()
        for x in dataloader:
            x = x.to(device)
            self.model.zero_grad()
            out, _ = self.model(x)
            loss = out.mean()
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2)

        for n in fisher:
            fisher[n] = fisher[n] / len(dataloader)
        self._fisher = fisher

    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self._fisher:
                loss += (self._fisher[n] * (p - self._means[n]) ** 2).sum()
        return self.lambda_ewc * loss


