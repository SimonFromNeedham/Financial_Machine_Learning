"""Evaluates (back-tests) the performance of a machine learning model created with svm_trainer in the stock market"""
from itertools import combinations
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from svm_trainer import Trainer


class Tester:
    def __init__(self):
        trainer = Trainer(20)
        data = trainer.get_data()

        # Split data into training and testing sets
        self.train, self.test = np.array_split(data, 2)

    def test_model(self, features):
        # Select features to be tested
        x_train = self.train[features]
        x_test = self.test[features]

        # Parse out targets from data
        y_train = self.train['class']
        y_test = self.test['class']

        # Train the machine learning model
        model = Trainer.train_model(x_train, y_train)

        # Evaluate the model's performance
        y_test_prediction = model.predict(x_test)
        print('\nFeatures:', features, 'Accuracy:', round(accuracy_score(y_test, y_test_prediction), 3))

        # Create, normalize, and display a confusion matrix
        y_test_prediction = pd.Series(list(y_test_prediction), name='Predicted')
        y_test = pd.Series(list(y_test), name='Actual')
        cm = pd.crosstab(y_test, y_test_prediction)
        print(cm/cm.sum(axis=0))


def main():
    tester = Tester()
    features = ['ADX', 'MACD', 'OBV', 'RSI']

    # Check feature combos of all lengths
    for num in range(1, len(features) + 1):
        for combo in combinations(features, num):
            tester.test_model(list(combo))


if __name__ == '__main__':
    main()
