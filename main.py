import fakeNewsNN
import torch
import constantNames
import os
from fakeNewsDataframe import fakeNewsDataframe
import getDevice
from torch.nn.utils import rnn
import pandas as pd
from fakeNewsDataset import fakeNewsDataset
from torch.utils.data import DataLoader
from collateFunc import collate_func

def main():
    if not os.path.exists(constantNames.fileName):
        print(f"No model saved at {constantNames.fileName}. Please train a model")
        return

    fndf = fakeNewsDataframe()
    
    model = fakeNewsNN.fakeNewsNN(fndf.vocabSize)
    model = model.to(getDevice.get_device_func())
    model.load_state_dict(torch.load(constantNames.fileName, weights_only=True))
    model.eval()
    
    continueLoop = True

    while continueLoop:
        title = input("Enter title of article: ")
        body = input("Enter body of article: ")
        titleAndBody = title + " " + body
        titleAndBody = fndf.convertToLower(titleAndBody)
        titleAndBody = fndf.getRidOfPunc(titleAndBody)
        titleAndBody = titleAndBody.split()
        titleAndBody = fndf.processSentence(titleAndBody)

        tabDataloader = {constantNames.titleAndTextName: [titleAndBody],
                         constantNames.truthfulnessName: [0]}
        tabDataloader = pd.DataFrame(tabDataloader)
        tabDataloader = fakeNewsDataset(tabDataloader)
        tabDataloader = DataLoader(tabDataloader, collate_fn=collate_func)

        for (sample, _) in tabDataloader:
            prediction = model(sample)
            prediction = prediction.squeeze()
            prediction = model.sigmoid(prediction)

            if prediction >= 0.5:
                print("The article is probably fake")
            else:
                print("The article is probably real")
        
        shouldContinue = input("Do another article? (Type 'yes' to do another article): ")
        shouldContinue = shouldContinue.lower()

        if shouldContinue != "yes":
            continueLoop = False

        

if __name__ == "__main__":
    main()