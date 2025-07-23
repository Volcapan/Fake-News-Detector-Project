import sklearn.model_selection
from fakeNewsDataframe import fakeNewsDataframe
from fakeNewsDataset import fakeNewsDataset
import sklearn
import torch

def main():
    randomStateVal = 440
    dfTest = fakeNewsDataframe()
    trainDataframe, tempDataframe = sklearn.model_selection.train_test_split(dfTest.news, test_size=0.4, random_state=randomStateVal)
    validateDataframe, testDataframe = sklearn.model_selection.train_test_split(tempDataframe, test_size=0.5, random_state=randomStateVal)

    trainDataset = fakeNewsDataset(trainDataframe)
    validateDataset = fakeNewsDataset(validateDataframe)
    testDataset = fakeNewsDataset(testDataframe)

    print(trainDataset.__len__())
    print(validateDataset.__len__())
    print(testDataset.__len__())


if __name__ == "__main__":
    main()