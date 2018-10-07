import scripts.wordbased_lstm_params as params
import pickle as pkl
import os
from shutil import copyfile

labels = pkl.load(open(params.path_to_labels, 'rb'))
flat_list = [item for sublist in labels for item in sublist]

labelCounter = {}
labelIndex = {}

sourcePath = "data/corpus_2018-08-19/"
for root, dirs, files in os.walk(sourcePath):
    for fileName in files:
        fileidx = fileName.split(".")[0]
        if(fileidx in labels.keys()):
            fileLabel = labels[fileidx][0]
            labelCounter[fileLabel] = labelCounter.get(fileLabel, 0) + 1
            labelIndex[fileLabel] = labelIndex.get(fileLabel, []) + [fileidx]

targetPath = "data/corpus_filtered/"
minCount = 20
for key in labelCounter.keys():
    if labelCounter[key] >= minCount:
        counter=0
        for i in range(len(labelIndex[key])):
            fileIdx = labelIndex[key][i]
            stats = os.stat(sourcePath+fileIdx+".txt")
            KB_size = stats.st_size/1024
            if KB_size<300 and KB_size>20:
                copyfile(sourcePath+fileIdx+".txt", targetPath+fileIdx+".txt")
                counter += 1
            if counter==minCount:
                break
