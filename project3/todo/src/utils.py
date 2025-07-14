import torch
import torch.nn as nn


class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layer model.
    No non-linear activation allowed.
    """
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        self.layer = nn.Linear(input_dim, 1)
        self.acti = nn.Sigmoid()

    def forward(self, x):
        return self.acti(self.layer(x)).reshape(x.shape[:1])

    def predict_class(self, x):
        with torch.no_grad():
            ypred_prob = self(x)
            ypred = ypred_prob.clone().detach()
            ypred[ypred < 0.5] = 0
            ypred[ypred >= 0.5] = 1
            return ypred_prob, ypred


def entropy_loss(outputs, targets, sample_weights):
    return torch.sum(- sample_weights * targets * torch.log2(outputs))
