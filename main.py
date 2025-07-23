import pandas as pd
from fakeNewsDataframe import fakeNewsDataframe
from fakeNewsDataset import fakeNewsDataset

def main():
    dfTest = fakeNewsDataframe()

    #print(test.news)
    #print(test.news["titleAndText"])

    dsTest = fakeNewsDataset(dfTest)

    print(dsTest.__len__())
    print(dsTest.__getitem__(440))

if __name__ == "__main__":
    main()