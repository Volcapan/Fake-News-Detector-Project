import sklearn.model_selection
from fakeNewsDataframe import fakeNewsDataframe
from fakeNewsDataset import fakeNewsDataset
import sklearn
import torch
from torch.utils.data import DataLoader
import collateFunc
from fakeNewsNN import fakeNewsNN
import getDevice

def train(model, dataLoader, lossFunc, optimFunc):
    print("Training")

    model.train()

    for (samples, labels) in dataLoader:
        predictions = model(samples)
        predictions = predictions.float()
        predictions = predictions.squeeze()
        labels = labels.float()
        labels = labels.squeeze()

        loss = lossFunc(predictions, labels)

        loss.backward()
        optimFunc.step()
        optimFunc.zero_grad()

def validate(model, dataLoader, lossFunc):
    print("Evaluating")

    model.eval()

    numSamples = len(dataLoader.dataset)
    numBatches = len(dataLoader)
    totalLoss = 0
    totalCorrect = 0

    for (samples, labels) in dataLoader:
        predictions = model(samples)
        predictions = predictions.squeeze()
        labels = labels.squeeze()
        totalLoss += lossFunc(predictions, labels).item()

        predictions = model.sigmoid(predictions)
        for idx in range(len(predictions)):
            if predictions[idx] >= 0.5:
                predictions[idx] = 1
            else:
                predictions[idx] = 0
        
        totalCorrect += (predictions == labels).type(torch.float).sum().item()
    
    print(f"Correctness: {totalCorrect / numSamples}, Loss: {totalLoss / numBatches}")

def main():
    randomStateVal = 440
    batchSize = 110
    epochs = 5
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
    fakeNewsNNModel = fakeNewsNNModel.to(getDevice.get_device_func())
    lossFunc = torch.nn.BCEWithLogitsLoss()
    optimFunc = torch.optim.Adam(fakeNewsNNModel.parameters(), lr=learningRate)

    for epoch in range(epochs):
        print(f"Epoch #{epoch + 1}")

        train(fakeNewsNNModel, trainDataLoader, lossFunc, optimFunc)
        validate(fakeNewsNNModel, validateDataLoader, lossFunc)

    userInput = input("Test model on test data? (Type 'yes' to test): ")
    userInput = userInput.lower()

    if userInput == "yes":
        validate(fakeNewsNNModel, testDataLoader, lossFunc)
    
    userInput = input("Save model? (Type 'yes' to save): ")
    userInput = userInput.lower()

    if userInput == "yes":
        torch.save(fakeNewsNNModel.state_dict(), "savedFakeNewsNNModel.pth")
        print("Model saved")


if __name__ == "__main__":
    main()