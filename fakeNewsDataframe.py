import pandas as pd
import numpy
import re

class fakeNewsDataframe:
    def convertToLower(_, aStr):
        return aStr.lower()
    
    def getRidOfPunc(_, aStr):
        return re.sub(r"[^\w\s]", "", aStr)
    
    def __init__(self, path = ""):
        if path == "":
            fakeNews = pd.read_csv("Data/Fake.csv")
            fakeNews["truthfulness"] = numpy.zeros(fakeNews.shape[0])

            realNews = pd.read_csv("Data/True.csv")
            realNews["truthfulness"] = numpy.ones(realNews.shape[0])
            
            self.news = pd.concat([fakeNews, realNews], ignore_index=True)
            self.news["title"] =  self.news["title"].apply(self.convertToLower)
            self.news["title"] = self.news["title"].apply(self.getRidOfPunc)
            self.news["title"] = self.news["title"].astype('string')
            self.news["text"] = self.news["text"].apply(self.convertToLower)
            self.news["text"] = self.news["text"].apply(self.getRidOfPunc)
            self.news["text"] = self.news["text"].astype('string')
        else:
            print("Functionality not built yet")