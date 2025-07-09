import pandas as pd
from fakeNewsDataframe import fakeNewsDataframe
from fakeNewsDataset import fakeNewsDataset

def main():
    test = fakeNewsDataframe()
    test2 = fakeNewsDataset(test.news)

    print(test2.__len__())

    testSample, testLabel = test2.__getitem__(index=0)

    print(testSample)
    print(testLabel)

if __name__ == "__main__":
    main()