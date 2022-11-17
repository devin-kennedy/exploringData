import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder


def main():
    penguin_data = pd.read_csv('./penguins.csv').dropna()

    xs = penguin_data[penguin_data["species"] == "Adelie"]["body_mass_g"]
    ys = OrdinalEncoder().fit_transform(penguin_data[penguin_data["species"] == "Adelie"].sex.values.reshape(-1, 1))

    X_train, X_test, Y_train, Y_test = train_test_split(
        xs,
        ys,
        random_state=2,
        train_size=0.7
    )
    lr = LogisticRegression(random_state=0).fit(X_train.values.reshape(-1, 1), Y_train)

    plt.scatter(xs, ys)
    line_xs = np.arange(3000, 5000, 100)
    logistic_y = lr.predict_proba(line_xs.reshape(-1, 1))
    plt.plot(line_xs, logistic_y[:, 1])
    plt.show()


if __name__ == '__main__':
    main()
