"""Trains an SVM ML model to predict changes in the stock market based on data gathered from Alpha Vantage."""
from random import choice
from random import sample
from time import sleep

from sklearn.svm import SVC

from generator import Generator


class Trainer:
    def __init__(self, num_securities):
        # Collect data on this many securities
        self.num_securities = num_securities-1

        # Compile securities
        etf = open('etf.txt')

        # Read the file into a list
        self.symbols = etf.readlines()

    def get_data(self):
        """Generate a DataFrame of random securities' historical data."""
        # Choose a random security as the base for the data set
        generator = Generator(choice(self.symbols)[:-1])

        # Don't print out data for simplicity
        data = generator.get_data(False)

        for symbol in sample(self.symbols, self.num_securities):
            # Print the program's progress to the console
            print('Collecting data for:', symbol[:-1])

            # Avoid overusing AV API (to be safe, 1 call / min)
            sleep(60)

            # Collect the security's historical data
            generator.set_symbol(symbol[:-1])
            data = data.append(generator.get_data(False))

        return data

    @staticmethod
    def train_model(x_train, y_train):
        """Train an SVM model."""
        svm = SVC(gamma=.5, C=.5)
        svm.fit(x_train, y_train)
        return svm


def main():
    trainer = Trainer(5)
    print(trainer.get_data())


if __name__ == '__main__':
    main()
