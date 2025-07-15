import pandas as pd
from fakeNewsDataframe import fakeNewsDataframe
from fakeNewsDataset import fakeNewsDataset

def main():
    test = fakeNewsDataframe()

    print(test.news)
    # test2 = fakeNewsDataset(test.news)

    # bprint(test2.__len__())

    # testSample, testLabel = test2.__getitem__(index=0)

    # print(testSample)
    # print(testLabel)

if __name__ == "__main__":
    main()