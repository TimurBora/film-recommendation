import pandas as pd


def read_movies_csv(path):
    df = pd.read_csv(path)
    print(df.head())
