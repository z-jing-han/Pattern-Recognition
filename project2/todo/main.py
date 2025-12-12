import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        inputs = np.insert(inputs, 0, values=np.ones((inputs.shape[0],)), axis=1)
        scale = np.std(np.transpose(inputs), ddof=1, axis=1)
        self.weights = np.array([np.random.normal(0, i) for i in scale])
        for i in range(self.num_iterations):
            targets_pred_probs = self.sigmoid(inputs @ np.transpose(self.weights))
            delta = (1 / inputs.shape[0]) * ((targets_pred_probs - targets) @ inputs)
            self.weights = self.weights - self.learning_rate * delta
        self.intercept = self.weights[0]
        self.weights = self.weights[1:]

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Tuple[t.Sequence[np.float64], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        prob = self.sigmoid(inputs @ self.weights + self.intercept)
        return prob, np.array(list(0 if prob[i] < 0.5 else 1 for i in range(prob.shape[0])))

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))


class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        inputs0 = inputs[np.where(targets == 0)]
        inputs1 = inputs[np.where(targets == 1)]
        self.m0 = np.mean(inputs0, axis=0).reshape(-1, 1)
        self.m1 = np.mean(inputs1, axis=0).reshape(-1, 1)
        self.sw = np.add(np.cov(inputs0, rowvar=False), np.cov(inputs1, rowvar=False))
        self.sb = np.dot((self.m1 - self.m0), np.transpose(self.m1 - self.m0))
        self.w = np.dot(np.linalg.inv(self.sw), (self.m1 - self.m0))
        self.w /= np.linalg.norm(self.w)

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Sequence[t.Union[int, bool]]:
        test_project = np.transpose(np.dot(inputs, self.w))[0]
        threshold = np.mean(test_project) + 1e-2
        return np.array([0 if x < threshold else 1 for x in test_project])

    def plot_projection(self, inputs: npt.NDArray[float]):
        inputs_project = np.dot(np.dot(inputs, self.w) / np.linalg.norm(self.w)**2, np.transpose(self.w))
        self.slope = self.w[1][0] / self.w[0][0]
        targets = self.predict(inputs)
        plt.axline(xy1=(0, 0), slope=self.slope, c='gray')
        for index, point in enumerate(inputs_project):
            color = 'r' if targets[index] else 'b'
            plt.plot([point[0], inputs[index, 0]], [point[1], inputs[index, 1]], ls=':', alpha=0.5, c=color)
        sns.scatterplot(x=inputs_project[:, 0], y=inputs_project[:, 1], hue=targets, alpha=0.5, legend=False)
        sns.scatterplot(x=inputs[:, 0], y=inputs[:, 1], hue=targets, alpha=0.5)
        plt.title(f'Projection line: slope = {self.slope:.4f}, intercept = 0')
        plt.gca().axis('square')
        plt.show()


def compute_auc(y_trues, y_preds) -> float:
    fpr, tpr, _ = metrics.roc_curve(y_trues, y_preds)
    return metrics.auc(fpr, tpr)


def accuracy_score(y_trues, y_preds) -> float:
    return 1 - sum(abs(y_trues - y_preds)) / y_trues.shape[0]


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=1.5e-2,  # You can modify the parameters as you want
        num_iterations=10000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    FLD_.fit(x_train, y_train)
    y_preds = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_preds)
    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')
    FLD_.plot_projection(x_test)


if __name__ == '__main__':
    main()
