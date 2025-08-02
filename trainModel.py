import sklearn.model_selection
from fakeNewsDataframe import fakeNewsDataframe
from fakeNewsDataset import fakeNewsDataset
import sklearn
import torch
from torch.utils.data import DataLoader
import collateFunc
from fakeNewsNN import fakeNewsNN

def train(model, dataLoader, lossFunc, optimFunc):
    print("In progress")

def main():
    randomStateVal = 440
    batchSize = 110
    epochs = 10
    learningRate = 1e-3
    
    fakeNewsDF = fakeNewsDataframe()
    trainDataframe, tempDataframe = sklearn.model_selection.train_test_split(fakeNewsDF.news, test_size=0.4, random_state=randomStateVal)
    validateDataframe, testDataframe = sklearn.model_selection.train_test_split(tempDataframe, test_size=0.5, random_state=randomStateVal)

    trainDataset = fakeNewsDataset(trainDataframe)
    validateDataset = fakeNewsDataset(validateDataframe)
    testDataset = fakeNewsDataset(testDataframe)

    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, collate_fn=collateFunc.collate_func, shuffle=True)
    validateDataLoader = DataLoader(validateDataset, batch_size=batchSize, collate_fn=collateFunc.collate_func, shuffle=True)
    testDataLoader = DataLoader(testDataset, batch_size=batchSize, collate_fn=collateFunc.collate_func, shuffle=True)

    fakeNewsNNModel = fakeNewsNN(fakeNewsDF.vocabSize)
    lossFunc = torch.nn.MSELoss()
    optimFunc = torch.optim.Adam(fakeNewsNNModel.parameters(), lr=learningRate)

    for epoch in range(epochs):
        print(f"Epoch #{epoch + 1}")

        train(fakeNewsNNModel, trainDataLoader, lossFunc, optimFunc)


if __name__ == "__main__":
    main()