import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model


def main():
    penguin_data = pd.read_csv('./penguins.csv')

    sns.swarmplot(
        data=penguin_data,
        y="sex",
        x="body_mass_g"
    )
    plt.show()


if __name__ == '__main__':
    main()