import pandas as pd
from fakeNewsDataframe import fakeNewsDataframe

def main():
    test = fakeNewsDataframe()

    print(test.news.head())
    print(test.news.dtypes)

if __name__ == "__main__":
    main()