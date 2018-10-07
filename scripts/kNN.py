import numpy as np
from .wordList import WordList
from nltk.tokenize import sent_tokenize, word_tokenize
import threading
import time
from .featureGenerator import FeatureGenerator
from scipy.spatial import distance_matrix
from .gutenbergCharDataset import GutenbergCharDataset
import scripts.wordbased_lstm_params as params
import copy
import os
import pickle as pkl
from .logger import Logger


class KNN:
    def __init__(self, pwList, pClasses=[], numThreads=1, distanceMetric="euclid", pfeatureMethod=0, pNumNearestNeighbors=10):
        self.numNearestNeighbors = pNumNearestNeighbors
        self.numOutputLabels = 1

        self.wList = pwList

        self.classes = None
        self.updateClasses(pClasses)


        self.featureVectors = None
        self.labels = None
        self.numLabels = 10

        self.threadSemaphore = threading.Semaphore(numThreads)
        self.featureGenerator = None
        self.featureMethod = pfeatureMethod

        self.distanceMetric = distanceMetric

    def updateClasses(self, pClasses):
        ppClasses = copy.deepcopy(pClasses)
        flat_list = [item for sublist in ppClasses for item in sublist]
        flat_list.append("other")
        if self.classes is None:
            self.classes = flat_list
        else:
            self.classes = self.classes + flat_list
        self.classes = sorted(list(set(self.classes)))
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
        self.featureGenerator = FeatureGenerator(self.wList, self.classes)
        return X,Y

    def train(self, X, Y, loadFromFile=False):
        assert len(X)==len(Y)
        if not loadFromFile:
            threadList = []
            for sampleIdx, fName in enumerate(X):
                th = threading.Thread(target=self.trainSingle, args=(fName, Y[sampleIdx], ))
                self.threadSemaphore.acquire()
                print("Training KNN on file "+str(sampleIdx+1)+"/"+str(len(X)))
                th.start()
                threadList.append(th)

            #wait until ALL thread have ended.
            for th in threadList:
                th.join()
            #save it
            pkl.dump(self.featureVectors, open("data/features-"+str(self.featureMethod)+".pkl", "wb+" ))
            pkl.dump(self.labels, open("data/labels-"+str(self.featureMethod)+".pkl", "wb+" ))
            pkl.dump(self.classes, open("data/classes-"+str(self.featureMethod)+".pkl", "wb+" ))
        else:
            self.featureVectors = pkl.load(open("data/features-"+str(self.featureMethod)+".pkl", "rb" ))
            self.labels = pkl.load(open("data/labels-"+str(self.featureMethod)+".pkl", "rb" ))
            self.classes = pkl.load(open("data/classes-"+str(self.featureMethod)+".pkl", "rb" ))
            self.updateClasses([self.classes])
            self.featureGenerator = FeatureGenerator(self.wList, self.classes)

    def trainSingle(self, fName, Y):
        X, Y = self.featureGenerator.generateBagOfWords(fName, Y, featureMethod=self.featureMethod)
        X = X[None,0]
        if self.featureVectors is None:
            self.featureVectors = X
            while Y.shape[0]<self.numLabels:
                Y = np.hstack((Y, -1))
            self.labels = Y
        else:
            self.featureVectors = np.vstack((self.featureVectors,X))
            while Y.shape[0]<self.numLabels:
                Y = np.hstack((Y, -1))
            self.labels = np.vstack((self.labels,Y))
        self.threadSemaphore.release()

    def predict(self, X, loadFromFile=True):
        predictionVector = np.zeros((len(X), len(self.wList)))
        predictions = np.ones((len(X), self.numOutputLabels)) * -1

        #generate th feature vectors. The labels are mapped!
        if not loadFromFile:
            for i, fName in enumerate(X):
                print("Generate feature vector for prediction "+str(i+1)+"/"+str(len(X)))
                predictionX, _ = self.featureGenerator.generateBagOfWords(fName, featureMethod=self.featureMethod)
                predictionVector[i,:] = predictionX[0,:]
            pkl.dump(predictionVector, open("data/featuresTest-"+str(self.featureMethod)+".pkl", "wb+" ))
        else:
            predictionVector = pkl.load(open("data/featuresTest-"+str(self.featureMethod)+".pkl", "rb" ))
        #search for the nearest neighbors and get their (already mapped) labels
        if self.distanceMetric=="cosine":
            predictionVector = predictionVector/np.linalg.norm(predictionVector, axis=-1)[:,None]
            self.featureVectors = self.featureVectors/np.linalg.norm(self.featureVectors, axis=-1)[:,None]
            distMat = -1*np.matmul(predictionVector, np.transpose(self.featureVectors))
        elif self.distanceMetric == "manhattan":
            distMat = distance_matrix(predictionVector, self.featureVectors, p=1)
        elif self.distanceMetric == "euclidean":
            distMat = distance_matrix(predictionVector, self.featureVectors, p=2)
        indices = np.argsort(distMat, axis=1)[:,0:self.numNearestNeighbors]
        temp_classes = np.asarray(self.classes)
        for i, idxs in enumerate(indices):
            nearestNeighborLabels = self.labels[idxs] + 1
            nearestNeighborLabels = nearestNeighborLabels.astype("int64")
            counts = np.bincount(nearestNeighborLabels.flatten())
            counts = counts[1:]
            sortIndices = np.flip(np.argsort(counts), axis=-1)
            predictions[i,:] = sortIndices[:self.numOutputLabels]
        predictions = predictions.astype("int64")

        #map labels back!
        finalPredictions = np.chararray(predictions.shape, itemsize=20)
        for row in range(predictions.shape[0]):
            for col in range(predictions.shape[1]):
                finalPredictions[row,col] = self.classes[predictions[row,col]]
        return finalPredictions

if __name__ == '__main__':
    loadFromFiles = True

    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--metric', type=str, default="euclidean", help='Distance metric for KNN')
    parser.add_argument('--fMethod', type=int, default=0, help='Distance metric for KNN')
    parser.add_argument('--numNeighbors', type=int, default=10, help='Distance metric for KNN')
    args = parser.parse_args()

    import sys
    sys.stdout = Logger("data/logs/kNN-"+args.metric+"-"+str(args.fMethod)+"-"+str(args.numNeighbors)+".log")

    #generate Wordlist
    wList = WordList()
    wList.loadWordList(path="data/wordListAll.pkl")
    print("Length of wordList: "+str(len(wList)))

    #generate data
    knn = KNN(wList, distanceMetric=args.metric, pfeatureMethod=args.fMethod, pNumNearestNeighbors=args.numNeighbors)
    trainX, trainY = knn.generateXandY(pPath=["data/train_corpus/"])
    testX, testY = knn.generateXandY(pPath=["data/test_corpus/"])
    if(loadFromFiles):
        testY = pkl.load(open("data/labelsTest-"+str(args.fMethod)+".pkl", "rb" ))
    else:
        pkl.dump(testY, open("data/labelsTest-"+str(args.fMethod)+".pkl", "wb+" ))

    #train & predict
    knn.train(trainX, trainY, loadFromFile=loadFromFiles)
    predLabels = knn.predict(testX, loadFromFile=loadFromFiles)

    #test accuracy
    accurate = np.zeros(predLabels.shape[0])
    for i, labels in enumerate(predLabels):
        for sublabel in labels:
            if sublabel.decode("utf-8") in testY[i]:
                accurate[i] = 1
    accuracy = float(np.sum(accurate))/float(accurate.shape[0])
    print(predLabels, testY)
    print("TP/#documents: "+ str(accuracy))

    #oneHot
    import torch
    oneHotPred = torch.zeros((len(testY),len(knn.classes)))
    oneHotReal = torch.zeros((len(testY),len(knn.classes)))
    for i in range(oneHotPred.shape[0]):
        idx = knn.classes.index(predLabels[i].decode("utf-8"))
        oneHotPred[i,idx] = 1
    for i in range(oneHotReal.shape[0]):
        for sublabel in testY[i]:
            try:
                idx = knn.classes.index(sublabel)
                oneHotReal[i,idx] = 1
            except:
                print("MISSED A LABEL")
    from .run_vanilla_NN import predictions_correct
    accuracy, precision, recall, true_positive, true_negative, false_positive, false_negative = predictions_correct(oneHotPred, oneHotReal)
    print("Accuracy:", accuracy, "; Precision:",precision,"; Recall:", recall)
