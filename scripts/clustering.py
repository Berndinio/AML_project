from sklearn.cluster import KMeans,DBSCAN
import numpy as np
from .featureGenerator import FeatureGenerator
from .wordList import WordList
import threading
import copy
import os
import pickle as pkl
import scripts.wordbased_lstm_params as params
from .logger import Logger


class Clustering:
    def __init__(self, pWordList, pClasses=[], numThreads=1, pfeatureMethod=0):
        self.wordList = pWordList

        #the class for crap labels
        self.classes = None
        self.updateClasses(pClasses)
        print("Possible labels:", self.classes)

        self.featureVectors = None
        self.labelVectors = None

        self.featureGenerator = None
        self.threadSemaphore = threading.Semaphore(numThreads)
        self.lock = threading.Lock()
        self.num_jobs = 4
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


    def generateFeatures(self, X, Y=None, loadFromFile=False, flag="train", pPath="data/"):
        if not loadFromFile:
            if Y is None:
                Y=[]
                for i in range(len(X)):
                    Y.append(["other"])
            threadList = []
            self.featureVectors = None
            for sampleIdx, fName in enumerate(X):
                th = threading.Thread(target=self.generateFeaturesSingle, args=(fName, Y[sampleIdx]))
                self.threadSemaphore.acquire()
                print("Generating features for clustering on file "+str(sampleIdx+1)+"/"+str(len(X)))
                th.start()
                threadList.append(th)

            #wait until ALL thread have ended.
            for th in threadList:
                th.join()
        else:
            if(flag=="train"):
                self.featureVectors = pkl.load(open(pPath+"features-"+str(self.featureMethod)+".pkl", "rb" ))
                self.labelVectors = pkl.load(open(pPath+"labels-"+str(self.featureMethod)+".pkl", "rb" ))[:,0,None]
                self.classes = pkl.load(open(pPath+"classes-"+str(self.featureMethod)+".pkl", "rb" ))
                self.updateClasses([self.classes])
                self.featureGenerator = FeatureGenerator(self.wordList , self.classes)
            else:
                self.featureVectors = pkl.load(open(pPath+"featuresTest-"+str(self.featureMethod)+".pkl", "rb" ))
        return self.featureVectors, self.labelVectors

    def generateFeaturesSingle(self, fName, p_Y):
        f = open(fName, 'r', encoding="utf-8")
        corpus = f.read()
        X, Y = self.featureGenerator.generateBagOfWords(fName, [p_Y[0]], featureMethod=self.featureMethod)
        self.lock.acquire()
        if(self.featureVectors is None):
            self.featureVectors = X
            self.labelVectors = Y
        else:
            self.featureVectors = np.vstack((self.featureVectors, X))
            self.labelVectors = np.vstack((self.labelVectors, Y))
        self.lock.release()
        self.threadSemaphore.release()



class kMeans(Clustering):
    def __init__(self, pWordList,  pClasses=[], pfeatureMethod=0):
        super().__init__(pWordList, pClasses, pfeatureMethod=pfeatureMethod)
        self.kMeansObj = None
        self.mapping = None

    def train(self, X_train, Y_train, loadFromFile=False, pPath="data/"):
        self.kMeansObj = KMeans(n_clusters=len(self.classes)-1, random_state=0, n_jobs=self.num_jobs)
        print("Training kmeans...")
        X_t, Y_t = self.generateFeatures(X_train, Y_train, loadFromFile=loadFromFile, flag="train", pPath=pPath)
        self.kMeansObj.fit(X_t)
        prediction = self.kMeansObj.predict(X_t)
        self.mapping = np.zeros((np.unique(prediction).shape[0]))
        for label in np.unique(prediction):
            idxs = np.where(prediction==label)[0]
            maxIdx = np.argmax(np.bincount(self.labelVectors[idxs,0].astype("int64")))
            self.mapping[label] = maxIdx

    def predict(self, X, loadFromFile=True):
        print("Predicting kmeans...")
        X_p, _ = self.generateFeatures(X, loadFromFile=loadFromFile, flag="test")
        prediction = self.kMeansObj.predict(X_p)
        finalPredictions = np.chararray(prediction.shape, itemsize=20)
        for i, label in enumerate(prediction):
            finalPredictions[i] = self.classes[int(self.mapping[int(prediction[i])])]
        return finalPredictions

class Dbscan(Clustering):
    def __init__(self, pWordList, pClasses=[], pmetric="euclidean", peps=0.5, pmin_samples=5, pfeatureMethod=0):
        super().__init__(pWordList, pClasses, pfeatureMethod=pfeatureMethod)
        self.mapping = None
        self.DBSCANObj = DBSCAN(metric=pmetric, eps=peps, min_samples=pmin_samples, n_jobs=self.num_jobs)

    def predict(self, X_train, Y_train, X_predict, pPath="data/", loadFromFile=True):
        print("Training DBSCAN...")
        #Y_train2 = np.asarray(Y_train).flatten()
        X_t, Y = self.generateFeatures(X_train, Y_train, loadFromFile=loadFromFile, flag="train", pPath=pPath)
        X_p, _ = self.generateFeatures(X_predict, loadFromFile=loadFromFile, flag="test")

        X = np.vstack((X_t,X_p))
        prediction = self.DBSCANObj.fit_predict(X)
        prediction_t = prediction[:X_t.shape[0]]
        prediction_p = prediction[X_t.shape[0]:]

        #get the mapping
        self.mapping = np.zeros((np.unique(prediction_t).shape[0]))
        for label in np.unique(prediction_t):
            idxs = np.where(prediction_t==label)[0]
            maxIdx = np.argmax(np.bincount(Y[idxs,0].astype("int64")))
            self.mapping[label] = maxIdx

        #map the to_predict data
        finalPredictions = np.chararray(prediction_p.shape, itemsize=20)
        for i, label in enumerate(prediction_p):
            finalPredictions[i] = self.classes[int(self.mapping[int(prediction_p[i])])]
        return finalPredictions

if __name__ == '__main__':
    loadFromFiles = True

    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--algorithm', type=str, default="kMeans", help='Clustering Algorithm to use.')
    parser.add_argument('--eps', type=float,  default=0.5, help='Clustering Algorithm to use.')
    parser.add_argument('--metric', type=str, default="euclidean", help='Distance metric for KNN')
    parser.add_argument('--minSamples', type=int, default=5, help='Distance metric for KNN')
    parser.add_argument('--fMethod', type=int, default=0, help='Distance metric for KNN')
    args = parser.parse_args()

    #generate Wordlist
    wList = WordList()
    wList.loadWordList(path="data/wordListAll.pkl")
    print("Length of wordList: "+str(len(wList)))

    if(args.algorithm=="kMeans"):
        import sys
        sys.stdout = Logger("data/logs/kMeans-"+str(args.fMethod)+".log")
        #generate data
        kMeansObj = kMeans(wList, pfeatureMethod=args.fMethod)
        trainX, trainY = kMeansObj.generateXandY(pPath=["data/train_corpus/"])
        testX, testY = kMeansObj.generateXandY(pPath=["data/test_corpus/"])
        if loadFromFiles:
            testY = pkl.load(open("data/labelsTest-"+str(args.fMethod)+".pkl", "rb" ))
        #train
        kMeansObj.train(trainX, trainY, loadFromFile=loadFromFiles)
        predLabels = kMeansObj.predict(testX, loadFromFile=loadFromFiles)
        #test accuracy
        accurate = np.zeros(predLabels.shape[0])
        for i, sublabel in enumerate(predLabels):
            if sublabel.decode("utf-8") in testY[i]:
                accurate[i] = 1
        accuracy = float(np.sum(accurate))/float(accurate.shape[0])
        print(predLabels, testY)
        print("TP/#documents: "+ str(accuracy))

        #oneHot
        import torch
        oneHotPred = torch.zeros((len(testY),len(kMeansObj.classes)))
        oneHotReal = torch.zeros((len(testY),len(kMeansObj.classes)))
        for i in range(oneHotPred.shape[0]):
            idx = kMeansObj.classes.index(predLabels[i].decode("utf-8"))
            oneHotPred[i,idx] = 1
        for i in range(oneHotReal.shape[0]):
            for sublabel in testY[i]:
                try:
                    idx = kMeansObj.classes.index(sublabel)
                    oneHotReal[i,idx] = 1
                except:
                    print("MISSED A LABEL")
        from .run_vanilla_NN import predictions_correct
        accuracy, precision, recall, true_positive, true_negative, false_positive, false_negative = predictions_correct(oneHotPred, oneHotReal)
        print("Accuracy:", accuracy, "; Precision:",precision,"; Recall:", recall)
    elif(args.algorithm=="DBSCAN"):
        import sys
        sys.stdout = Logger("data/logs/DBSCAN-"+str(args.eps)+"-"+args.metric+"-"+str(args.fMethod)+"-"+str(args.minSamples)+".log")

        #generate data
        DbscanObj = Dbscan(wList, pmetric=args.metric, peps=args.eps, pfeatureMethod=args.fMethod, pmin_samples=args.minSamples)
        trainX, trainY = DbscanObj.generateXandY(pPath=["data/train_corpus/"])
        testX, testY = DbscanObj.generateXandY(pPath=["data/test_corpus/"])
        if loadFromFiles:
            testY = pkl.load(open("data/labelsTest-"+str(args.fMethod)+".pkl", "rb" ))
        #trainPredict
        predLabels = DbscanObj.predict(trainX, trainY, testX, loadFromFile=True)
        #test accuracy
        accurate = np.zeros(predLabels.shape[0])
        for i, sublabel in enumerate(predLabels):
            if sublabel.decode("utf-8") in testY[i]:
                accurate[i] = 1
        accuracy = float(np.sum(accurate))/float(accurate.shape[0])
        print(predLabels, testY)
        print("TP/#documents: "+ str(accuracy))

        #oneHot
        import torch
        oneHotPred = torch.zeros((len(testY),len(DbscanObj.classes)))
        oneHotReal = torch.zeros((len(testY),len(DbscanObj.classes)))
        for i in range(oneHotPred.shape[0]):
            idx = DbscanObj.classes.index(predLabels[i].decode("utf-8"))
            oneHotPred[i,idx] = 1
        for i in range(oneHotReal.shape[0]):
            for sublabel in testY[i]:
                try:
                    idx = DbscanObj.classes.index(sublabel)
                    oneHotReal[i,idx] = 1
                except:
                    print("MISSED A LABEL")
        from .run_vanilla_NN import predictions_correct
        accuracy, precision, recall, true_positive, true_negative, false_positive, false_negative = predictions_correct(oneHotPred, oneHotReal)
        print("Accuracy:", accuracy, "; Precision:",precision,"; Recall:", recall)
