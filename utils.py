import matplotlib.pyplot as plt
import seaborn as sns


def plot_kl_distribution(df, title, filename):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='KL_grade')
    plt.title(title)
    plt.savefig(filename)
    plt.show()
