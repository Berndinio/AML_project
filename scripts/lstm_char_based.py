# Wir haben versucht einen Baeren draufzuwerfen, damit es funktioniert #RussianHacker
#    :"'._..---.._.'";
#    `.             .'
#    .'    ^   ^    `.
#   :      a   a      :                 __....._
#   :     _.-0-._     :---'""'"-....--'"        '.
#    :  .'   :   `.  :                          `,`.
#     `.: '--'--' :.'                             ; ;
#      : `._`-'_.'                                ;.'
#      `.   '"'                                   ;
#       `.               '                        ;
#        `.     `        :           `            ;
#         .`.    ;       ;           :           ;
#       .'    `-.'      ;            :          ;`.
#   __.'      .'      .'              :        ;   `.
# .'      __.'      .'`--..__      _._.'      ;      ;
# `......'        .'         `'""'`.'        ;......-'
#       `.......-'                 `........'



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .gutenbergCharDataset import GutenbergCharDataset
from torch.utils.data import Dataset, DataLoader
from .charList import CharList

if torch.cuda.is_available():
    cuda2 = torch.device('cuda:2')
else:
    cuda2 = torch.device('cpu')

class LSTMClassificationNet(nn.Module):

    def __init__(self, pNum_classes):
        super(LSTMClassificationNet, self).__init__()
        self.D = 128
        self.num_classes = pNum_classes
        #embedding layer
        self.embedding = nn.Linear(len(CharList()), 8)
        #conv blocks
        self.conv1 = nn.Conv1d(8, self.D, 5, padding=2)
        self.pool1 = nn.MaxPool1d(2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(self.D, self.D, 5, padding=2)
        self.pool2 = nn.MaxPool1d(2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(self.D, self.D, 3, padding=1)
        self.pool3 = nn.MaxPool1d(2, stride=2, padding=0)
        self.conv4 = nn.Conv1d(self.D, self.D, 3, padding=1)
        self.pool4 = nn.MaxPool1d(2, stride=2, padding=0)
        self.conv5 = nn.Conv1d(self.D, self.D, 3, padding=1)
        self.pool5 = nn.MaxPool1d(2, stride=2, padding=0)
        self.drop1 = nn.Dropout(0.5)
        #lstm layer
        self.lstm1 = nn.LSTM(input_size=self.D, hidden_size=self.D, bidirectional=True)
        self.drop2 = nn.Dropout(0.5)
        #final classification
        #self.classification = nn.Linear(16000, self.num_classes)
        self.classification = nn.Linear(2*self.D, self.num_classes)

    def forward(self, x):
        dims = x.shape
        #embedding layer
        #print(x.shape, "before embedding")
        x = self.embedding(x)
        #print(x.shape, "embedding")
        x = x.permute(0,2,1)
        #print(x.shape, "permute")
        #conv blocks
        x = self.pool1(F.relu(self.conv1(x)))
        #print(x.shape, "conv1")
        x = self.pool2(F.relu(self.conv2(x)))
        #print(x.shape, "conv2")
        #x = self.pool3(F.relu(self.conv3(x)))
        #x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.drop1(x)

        #lstm layer
        #print(x.shape)
        x = x.permute(2,0,1)
        out, h_n = self.lstm1(x)

        x = torch.cat((h_n[0][0], h_n[0][1]), 1)
        x = self.drop2(x)

        #x = x.view(x.shape[0],x.shape[1]*x.shape[2])
        #final classification
        x = F.softmax(self.classification(x), dim=1)
        indices = torch.argmax(x, dim=1, keepdim=True)
        return x, indices


def load_types(path):
    with open(path, 'r') as file:
        types = file.readlines()
    types = [t.rstrip("\n") for t in types]
    types.append("other")
    return types

if __name__ == '__main__':
    from .logger import Logger
    import sys
    sys.stdout = Logger("data/lastm_char_based.log")

    batch_size = 128
    num_epochs = 10
    #datasets
    train_dataset = GutenbergCharDataset(directory="data/train_corpus/")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataset = GutenbergCharDataset(directory="data/test_corpus/")
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=16)

    #create relevant Objects
    model = LSTMClassificationNet(len(train_dataset.uniqueLabels)).to(cuda2)
    lossObject = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), rho=0.9, eps=0.00001, weight_decay=0.0005/2.0)
    #optimizer = optim.Adadelta(model.parameters(), rho=0.9, eps=0.000001, weight_decay=0)
    #optimizer = optim.SGD(model.parameters(), lr=0.2)
    for epoch in range(num_epochs):
        #train
        model.train()
        print("Epoch "+str(epoch+1)+"/"+str(num_epochs))
        for i_batch, (sampledX, sampledY) in enumerate(train_dataloader):
            sampledX, sampledY = sampledX.to(cuda2), sampledY.to(cuda2)
            optimizer.zero_grad()
            out, output = model(sampledX)
            loss = lossObject(out.type(torch.cuda.FloatTensor), sampledY.type(torch.cuda.LongTensor)[:,0])
            print("Trainbatch "+str(i_batch+1)+"/"+str(len(train_dataloader))+" Loss:"+str(loss.cpu().data) )
            loss.backward()
            optimizer.step()
        torch.save(model, "data/charLSTM-"+str(epoch)+".pt")

        #test
        model.eval()
        #overallLoss = 0
        correct = 0
        num = 0
        print("Testbatch ")
        for i_batch, (sampledX, sampledY) in enumerate(test_dataloader):
            sampledX, sampledY = sampledX.to(cuda2), sampledY.to(cuda2)
            print(str(i_batch+1)+"/"+str(len(test_dataloader)), end=" ")
            out, output = model(sampledX)
            #print(sampledY,output)
            correct += torch.sum(sampledY.type(torch.cuda.FloatTensor)==output.type(torch.cuda.FloatTensor))
            num += sampledY.shape[0]
        print(" ")
        print("TestCorrect" , correct, num, correct/num)
