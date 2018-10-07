import numpy as np
from .wordList import WordList
from nltk.tokenize import sent_tokenize, word_tokenize
import threading
import time
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse

class FeatureGenerator:
    def __init__(self,pWordList, pClasses):
        self.wordList = pWordList
        #the class for crap labels
        import copy
        ppClasses = copy.deepcopy(pClasses)
        ppClasses.append('other')
        #flat_list = [item for sublist in ppClasses for item in sublist]
        self.classes = sorted(list(set(ppClasses)))

    def generateBagOfWords(self, fName, Y=["other"], featureMethod=0):
        """Short summary.

        Parameters
        ----------
        fName : type String
            The path to the file which should be processed.
        Y : type List
            The labels as list..
        featureMethod : type int
            Bag of words capped to 1.0 or endlessly

        Returns
        -------
        type
            The feature and label matrix/vector.

        """
        f = open(fName, 'r', encoding="utf-8")
        corpus = f.read()

        #predictionX = sparse.dok_matrix((len(Y), len(self.wordList)))
        predictionX = np.zeros((len(Y), len(self.wordList)))
        predictionY = np.ones((len(Y))) * -1

        tokenized = self.wordList.prefilter(corpus)
        for word in tokenized:
            word = self.wordList.postfilter(word)
            try:
                wordIdx = self.wordList.index(word)
                predictionX[:,wordIdx] = predictionX[:,wordIdx] + 1.0
            except:
                print("SOME DAMN WORD WAS NOT FOUND!")
        for i,y in enumerate(Y):
            predictionY[i] = self.classes.index(y)

        if featureMethod==0:
            predictionX = np.where(predictionX>1, 1, 0)
        if featureMethod==1:
            pass
            #just do nothing
        return predictionX, predictionY

    def generate_LSTM_indexed(self, fName, Y=[-1]):
        f = open(fName, 'r', encoding="utf-8")
        corpus = f.read()

        splitted = self.wordList.preprefilter(corpus)
        endList = []
        for word in splitted:
            filtered = self.wordList.postfilter(word)
            if not filtered == "":
                endList.append(filtered)
        predictionX = np.zeros((len(Y), len(endList)))
        predictionY = np.ones((len(Y))) * -1
        for i in range(len(endList)):
            predictionX[:,i] = self.wordList.index(endList[i])
        for i,y in enumerate(Y):
            predictionY[i] = self.classes.index(y)
        return predictionX, predictionY

if __name__ == '__main__':
    wList = WordList()
    wList.generateWordList(restriction=[11,11])
    X = wList.processedFiles
    Y = [[1,2]]

    print("Length of wordList:"+str(len(wList)))
    fGen = FeatureGenerator(wList, Y)
    predictionX, predictionY = fGen.generate_LSTM_indexed(X[0],Y[0])
    print(predictionX)
    print(predictionY)
