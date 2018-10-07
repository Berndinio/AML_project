import numpy as np
from .wordList import WordList
from nltk.tokenize import sent_tokenize, word_tokenize
import threading
import time
from sklearn.naive_bayes import MultinomialNB
from .featureGenerator import FeatureGenerator
import copy
import os
import pickle as pkl
import scripts.wordbased_lstm_params as params
from .logger import Logger

class NaiveBayes:
    def __init__(self, pWordList, pClasses=[], numThreads=1, pfeatureMethod=0):
        """Constructor.

        Parameters
        ----------
        pClasses : list
            List of possible labels. Does not have to be unique, but a complete list.
        pWordList : WordList
        """
        self.wordList = pWordList

        #the class for crap labels
        self.classes = None
        self.updateClasses(pClasses)
        print("Possible labels:", self.classes)

        #number of Threads being used
        self.threadSemaphore = threading.Semaphore(numThreads)
        self.clf = MultinomialNB()

        self.featureGenerator = None
        self.featureMethod = pfeatureMethod

    def updateClasses(self, pClasses):
        ppClasses = copy.deepcopy(pClasses)
        flat_list = [item for sublist in ppClasses for item in sublist]
        flat_list.append("other")
        if self.classes is None:
            self.classes = flat_list
        else:
            self.classes = self.classes + flat_list

        self.classes = sorted(list(set(self.classes)))
        self.featureGenerator = FeatureGenerator(self.wordList, self.classes)
        print("Possible labels - KNN:", self.classes)

    def generateXandY(self, pPath):
        X = []
        for path in pPath:
            for root, dirs, files in os.walk(path):
                for fileName in files:
                    X.append(path+fileName)

        labels = pkl.load(open(params.path_to_labels, 'rb'))
        Y = []
        for i, fileName in enumerate(files):
            if fileName.split(".")[0] in labels.keys():
                Y.append(labels[fileName.split(".")[0]])
            else:
                X.pop(i)
        self.updateClasses(Y)
        self.featureGenerator = FeatureGenerator(self.wordList, self.classes)
        return X,Y

    def train_partial(self,X,Y, loadFromFile=True, pPath="data/"):
        assert len(X) == len(Y)
        """A class to train a Naive Bayes.

        Parameters
        ----------
        X : list
            List of filenames to analyze.
        Y : list
            List of classes per corpus.
        """
        if not loadFromFile:
            threadList = []
            for sampleIdx, fName in enumerate(X):
                th = threading.Thread(target=self.trainSingle, args=(fName, Y[sampleIdx], ))
                self.threadSemaphore.acquire()
                print("Training naive bayes on file "+str(sampleIdx+1)+"/"+str(len(X)))
                th.start()
                threadList.append(th)

            #wait until ALL thread have ended.
            for th in threadList:
                th.join()
        else:
            X = pkl.load(open(pPath+"features-"+str(self.featureMethod)+".pkl", "rb" ))
            Y = pkl.load(open(pPath+"labels-"+str(self.featureMethod)+".pkl", "rb" ))
            self.classes = pkl.load(open(pPath+"classes-"+str(self.featureMethod)+".pkl", "rb" ))
            self.updateClasses([self.classes])
            self.featureGenerator = FeatureGenerator(self.wordList , self.classes)
            self.clf.partial_fit(X, Y[:,0].astype("uint32"), classes=self.classes)

    def trainSingle(self, fName, Y):
        f = open(fName, 'r', encoding="utf-8")
        corpus = f.read()
        X, Y = self.featureGenerator.generateBagOfWords(fName, Y=Y, featureMethod=self.featureMethod)
        self.clf.partial_fit(X, Y.astype("uint32"), classes=self.classes)
        self.threadSemaphore.release()

    def predict(self,X, loadFromFile=True):
        """Short summary.

        Parameters
        ----------
        X : list of filenames
            Filenames which should be processed

        Returns
        -------
        Genres
            Indices of class labels according to self.classes.
            This is a priority list.
        """
        predictionVector = np.zeros((len(X), len(self.wordList)))
        if not loadFromFile:
            for i, fName in enumerate(X):
                X, _ = self.featureGenerator.generateBagOfWords(fName, featureMethod=self.featureMethod)
                predictionVector[i,:] = X
        else:
            predictionVector = pkl.load(open("data/featuresTest-"+str(self.featureMethod)+".pkl", "rb" ))
        idxs = self.clf.predict(predictionVector)
        #predictions = np.chararray((predictionVector.shape[0],1), itemsize=20)
        #for i, idx in enumerate(idxs):
        #    print(idx)
        #    print(self.classes[int(idx)])
        #    predictions[i,0] = self.classes[int(idx)]
        return idxs

if __name__ == '__main__':
    loadFromFiles = True

    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--fMethod', type=int, default=0, help='Distance metric for KNN')
    args = parser.parse_args()

    import sys
    sys.stdout = Logger("data/logs/naive_bayes_lib"+str(args.fMethod)+".log")

    #generate Wordlist
    wList = WordList()
    wList.loadWordList(path="data/wordListAll.pkl")
    print("Length of wordList: "+str(len(wList)))



    nBayes = NaiveBayes(wList, pfeatureMethod=args.fMethod)
    trainX, trainY = nBayes.generateXandY(pPath=["data/train_corpus/"])
    testX, testY = nBayes.generateXandY(pPath=["data/test_corpus/"])
    if loadFromFiles:
        testY = pkl.load(open("data/labelsTest-"+str(args.fMethod)+".pkl", "rb"))
    nBayes.train_partial(trainX, trainY, loadFromFile=loadFromFiles)

    predLabels = nBayes.predict(testX, loadFromFile=loadFromFiles)
    print(predLabels, testY)
    #test accuracy
    accurate = np.zeros(predLabels.shape[0])
    for i, sublabel in enumerate(predLabels):
        if sublabel in testY[i]:
            accurate[i] = 1
    accuracy = float(np.sum(accurate))/float(accurate.shape[0])
    print("TP/#documents: "+ str(accuracy))

    #oneHot
    import torch
    oneHotPred = torch.zeros((len(testY),len(nBayes.classes)))
    oneHotReal = torch.zeros((len(testY),len(nBayes.classes)))
    for i in range(oneHotPred.shape[0]):
        idx = nBayes.classes.index(predLabels[i])
        oneHotPred[i,idx] = 1
    for i in range(oneHotReal.shape[0]):
        for sublabel in testY[i]:
            try:
                idx = nBayes.classes.index(sublabel)
                oneHotReal[i,idx] = 1
            except:
                print("MISSED A LABEL")
    from .run_vanilla_NN import predictions_correct
    accuracy, precision, recall, true_positive, true_negative, false_positive, false_negative = predictions_correct(oneHotPred, oneHotReal)
    print("Accuracy:", accuracy, "; Precision:",precision,"; Recall:", recall)
