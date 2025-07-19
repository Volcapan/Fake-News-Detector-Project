import pandas as pd
import numpy
import re

class fakeNewsDataframe:
    def convertToLower(_, aStr):
        return aStr.lower()
    
    def getRidOfPunc(_, aStr):
        return re.sub(r"[^\w\s]", "", aStr)
    
    def processTitleAndText(self):
        numRows = len(self.news["title"])
        listOfTitleAndText = []
        vocabulary = set()
        self.wordMap = {}
        listOfProcessedTaT = []

        for index in range(numRows):
            titleAndText = self.news["title"].iloc[index] + " " + self.news["text"].iloc[index]
            titleAndText = titleAndText.split()
            listOfTitleAndText.append(titleAndText)
        
        for sentence in listOfTitleAndText:
            for word in sentence:
                vocabulary.add(word)
        
        vocabulary = list(vocabulary)

        for index, word in enumerate(vocabulary, start=2):
            self.wordMap[word] = index
        
        self.wordMap["<pad>"] = 0
        self.wordMap["<unknown>"] = 1

        for sentence in listOfTitleAndText:
            processedTaT = []

            for word in sentence:
                processedTaT.append(self.wordMap[word])
            
            listOfProcessedTaT.append(processedTaT)
        
        return listOfProcessedTaT
    
    def __init__(self):
        fakeNews = pd.read_csv("Data/Fake.csv")
        # truthfulness == 0 means fake
        fakeNews["truthfulness"] = numpy.zeros(fakeNews.shape[0])

        realNews = pd.read_csv("Data/True.csv")
        # truthfulness == 1 means real
        realNews["truthfulness"] = numpy.ones(realNews.shape[0])
            
        self.news = pd.concat([fakeNews, realNews], ignore_index=True)
        self.news["title"] =  self.news["title"].apply(self.convertToLower)
        self.news["title"] = self.news["title"].apply(self.getRidOfPunc)
        self.news["title"] = self.news["title"].astype('string')
        self.news["text"] = self.news["text"].apply(self.convertToLower)
        self.news["text"] = self.news["text"].apply(self.getRidOfPunc)
        self.news["text"] = self.news["text"].astype('string')

        self.news["titleAndText"] = self.processTitleAndText()

        self.news = self.news.drop("subject", axis=1)
        self.news = self.news.drop("date", axis=1)