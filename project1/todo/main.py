import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        # add basis(1) in first column of Matrix
        X = np.insert(X, 0, values=np.ones((X.shape[0],)), axis=1)
        self.weights = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
        self.intercept = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
        return X @ self.weights + self.intercept


class LinearRegressionGradientdescent(LinearRegressionBase):
    def __init__(self):
        LinearRegressionBase.__init__(self)
        self.losses = []

    def fit(self, X, y, learning_rate: float = 1e-4, epochs: int = 500, batch_size: int = 5,
            L1: bool = False, L1Rate: float = 1e-4):
        # add basis(1) in first column of Matrix
        X = np.insert(X, 0, values=np.ones((X.shape[0],)), axis=1)
        # init weight
        scale = np.std(np.transpose(X), ddof=1, axis=1)
        self.weights = np.array([np.random.normal(0, i) for i in scale])
        # column 1 is always same, reinit by other weight
        self.weights[0] = np.random.normal(0, sum(scale) / scale.shape[0])
        for i in range(epochs):
            batch_loss = []
            for j in range(int(X.shape[0] / batch_size)):
                # predict
                batchIndex = list(range(j * batch_size, (j + 1) * batch_size))
                y_pred = X[batchIndex] @ np.transpose(self.weights)
                # compute loss
                batch_loss.append(compute_mse(y[batchIndex], y_pred))
                # update weight
                delta = (-2 / batch_size) * ((y[batchIndex] - y_pred) @ X[batchIndex])
                if L1:
                    delta = np.array([delta[i] + L1Rate if self.weights[i] > 0 else delta[i] - L1Rate
                                     for i in range(X.shape[1])])
                self.weights = self.weights - learning_rate * delta
            self.losses.append(sum(batch_loss) / len(batch_loss))
        self.intercept = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
        return X @ self.weights + self.intercept

    def plot_learning_curve(self):
        self.losses = self.losses[1:]
        sns.lineplot(x=np.arange(len(self.losses[1:])), y=self.losses[1:], alpha=0.8, label='Training MSE loss')
        plt.title('Training loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE loss')
        plt.show()


def compute_mse(prediction, ground_truth):
    return sum((ground_truth - prediction) ** 2) / len(prediction)


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    LR_GD.fit(train_x, train_y, L1=True)
    LR_GD.plot_learning_curve()
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).sum()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = ((mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
