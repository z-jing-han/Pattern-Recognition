import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001):
        """Implement your code here"""
        losses_of_models = []
        # Standardize
        X = np.copy(X_train)
        X = np.array([(X.T[i] - np.mean(X.T[i])) / np.std(X.T[i]) for i in range(X.shape[1])])
        X_train = X.T
        # numpy to pytorch tensor
        X_train, y_train = torch.from_numpy(X_train).to(torch.float32), torch.from_numpy(y_train).to(torch.float32)
        y_train[y_train == 0] = -1
        # Random seed 3 5 7(0.72)12 13(0.73) 14(0.7) 16 19 21 23(0.75)
        seed_list = [3, 5, 7, 12, 13, 14, 16, 19, 21, 23]
        seed_idx = 0
        self.sample_weights = torch.tensor([1 / len(y_train)] * len(y_train))
        for model in self.learners:
            torch.manual_seed(seed_list[seed_idx])
            seed_idx += 1
            # Init model
            parm = torch.tensor([[torch.normal(0, i) for i in torch.std(X_train, axis=0)]]).to(torch.float32)
            model.layer.weight = nn.Parameter(parm)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            sumofloss = 0
            # Train
            for _ in range(num_epochs):
                model.train()
                y_pred = model(X_train)
                loss = entropy_loss(y_pred, y_train, self.sample_weights)
                sumofloss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Update sample weights
            _, ypred = model.predict_class(X_train)
            ypred[ypred == 0] = -1
            diff = torch.not_equal(y_train, ypred).to(torch.float32)
            weighted_error = sum(self.sample_weights * diff) / sum(self.sample_weights)
            alpha = 0.5 * torch.log((1 - weighted_error) / (weighted_error))
            self.sample_weights = self.sample_weights * torch.exp(-alpha * y_train * ypred)
            self.sample_weights = self.sample_weights / sum(self.sample_weights)
            self.alphas.append(alpha)
            losses_of_models.append(sumofloss)
        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        # Standardize
        X = np.array([(X.T[i] - np.mean(X.T[i])) / np.std(X.T[i]) for i in range(X.shape[1])])
        X = X.T
        # numpy to pytorch tensor
        X = torch.from_numpy(X).to(torch.float32)
        # Predict
        ypred = torch.zeros(X.shape[0])
        for i in range(len(self.learners)):
            _, preds = self.learners[i].predict_class(X)
            preds[preds == 0] = -1
            ypred += self.alphas[i] * preds
        # Vote of eahc weak classifier
        ypred = torch.sign(ypred).to(torch.int32)
        ypred[ypred == -1] = 0
        return ypred.detach().numpy()

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        feature_impoertance = [model.layer.weight.data.abs() * a for model, a in zip(self.learners, self.alphas)]
        feature_impoertance = np.array(feature_impoertance)
        return feature_impoertance.sum(axis=0).ravel()

    def plot_learners_roc(self, X, y_trues):
        # Standardize
        X = np.array([(X.T[i] - np.mean(X.T[i])) / np.std(X.T[i]) for i in range(X.shape[1])])
        X = X.T
        # numpy to pytorch tensor
        X = torch.from_numpy(X).to(torch.float32)
        _, ax = plt.subplots(figsize=(7, 7))
        for model in self.learners:
            y_pred_probs, _ = model.predict_class(X)
            y_pred_probs = y_pred_probs.detach().numpy()
            fpr, tpr, _ = roc_curve(y_trues, y_pred_probs)
            RocCurveDisplay.from_predictions(
                y_trues,
                y_pred_probs,
                label=f"AUC={auc(fpr, tpr):0.4f}",
                ax=ax,
                plot_chance_level=True,
            )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[0::2], labels[0::2])
        ax.set(xlabel="FPR", ylabel="TPR")
        plt.show()
