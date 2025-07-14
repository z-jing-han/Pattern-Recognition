import pandas as pd
from loguru import logger
import numpy as np

from src import AdaBoostClassifier, BaggingClassifier, DecisionTree, entropy, gini
import matplotlib.pyplot as plt


def main():
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    feature_names = list(train_df.drop(['target'], axis=1).columns)

    """
    Feel free to modify the following section if you need.
    Remember to print out logs with loguru.
    """

    # AdaBoost
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_adaboost.fit(
        X_train,
        y_train,
        num_epochs=500,
        learning_rate=0.001,
    )
    y_pred_classes = clf_adaboost.predict_learners(X_test)
    accuracy_ = 1 - sum(abs(y_test - y_pred_classes)) / len(y_test)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')

    clf_adaboost.plot_learners_roc(
        X=X_test,
        y_trues=y_test
    )

    feature_importance = clf_adaboost.compute_feature_importance()
    plt.title("Feature Importance ")
    plt.barh(feature_names, feature_importance)
    plt.show()

    # Bagging
    clf_bagging = BaggingClassifier(
        input_dim=X_train.shape[1],
    )

    _ = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=500,
        learning_rate=0.001,
    )
    y_pred_classes = clf_bagging.predict_learners(X_test)
    accuracy_ = 1 - sum(abs(y_test - y_pred_classes)) / len(y_test)
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')

    clf_bagging.plot_learners_roc(
        X=X_test,
        y_trues=y_test
    )

    feature_importance = clf_bagging.compute_feature_importance()
    plt.title("Feature Importance ")
    plt.barh(feature_names, feature_importance)
    plt.show()

    # Decision Tree
    test = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    logger.info(f'gini index: {gini(test):.4f}')
    logger.info(f'entropy: {entropy(test):.4f}')

    clf_tree = DecisionTree(
        max_depth=7,
    )
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = 1 - sum(abs(y_test - y_pred_classes)) / len(y_test)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')

    plt.title("Feature Importance ")
    plt.barh(feature_names, clf_tree.feature_importance)
    plt.show()


if __name__ == '__main__':
    main()
