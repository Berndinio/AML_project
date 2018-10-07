import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from .charList import CharList
import pickle as pkl
import scripts.wordbased_lstm_params as params

class GutenbergCharDataset(Dataset):

    def __init__(self, directory="data/test_corpus/", pSequenceLength=3000, toNp=False):
        #Y
        self.labels = pkl.load(open(params.path_to_labels, 'rb'))
        self.uniqueLabels = self.load_types(params.path_to_genre)

        #X
        self.filelist = []
        for root, dirs, files in os.walk(directory):
            for fileName in files:
                if fileName.split(".")[0] in self.labels.keys():
                    if not self.labels[fileName.split(".")[0]][0] in []:
                        self.filelist.append(fileName)
        self.root_dir = directory
        self.sequenceLength = pSequenceLength


        self.toNP = toNp
        self.charList = CharList()
        self.numSplits = 20

    def load_types(self, path):
        with open(path, 'r') as file:
            types = file.readlines()
        types = [t.rstrip("\n") for t in types]
        types.append("other")
        return sorted(types)

    def __len__(self):
        return len(self.filelist)*self.numSplits

    def __getitem__(self, idx):
        fileIdx = int(idx/self.numSplits)
        fileOffset = idx%self.numSplits
        filename = self.filelist[fileIdx]
        f = open(self.root_dir+filename, 'r', encoding="utf-8")
        finalText = []
        corpus = f.read()
        #read characters
        calculatedIdx = (fileOffset*self.sequenceLength)%len(corpus)
        timeSeries = torch.zeros(self.sequenceLength, len(self.charList))
        for i, pChar in enumerate(corpus[calculatedIdx:calculatedIdx+self.sequenceLength]):
            fIdx = self.charList.index(pChar)
            if fIdx!=-1:
                timeSeries[i, fIdx] = 1

        fileIndexLookup = filename.split(".")[0]
        label = self.labels[fileIndexLookup][0]
        labelInt = self.uniqueLabels.index(label)
        labels = torch.ones(1)
        labels[0] = labelInt
        if self.toNP:
            return timeSeries.numpy(), labels.numpy()
        return timeSeries, labels
