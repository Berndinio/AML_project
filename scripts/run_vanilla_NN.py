import math
import pickle
import random
import time
import torch
import torch.nn as nn
from os import walk, path

import scripts.vanilla_NN_params as params  # import network parameters from separate file.
from scripts.wordList import WordList
from scripts.featureGenerator import FeatureGenerator

if torch.cuda.is_available():
    deviceCC = torch.device('cuda:0')
else:
    deviceCC = torch.device('cpu')

# randomly initialize training, validation and test set, given the percentages of validation and test data
def create_training_validation_test_set(path_to_corpus, validation_percentage, test_percentage):
    files = []
    for(_, _, filenames) in walk(path_to_corpus):
        files.extend(filenames)
        break
    files_in_corpus = len(files)  # the corpus may not contain all files in [x,y]

    # number of elements in validation and test set (10% and 1% of the corpus respectively)
    num_validation_and_test = int(math.ceil(files_in_corpus*(validation_percentage+test_percentage)))
    validation_and_test_files = []
    while len(validation_and_test_files) <= num_validation_and_test+1:
        choice = random.choice(files)
        if choice not in validation_and_test_files:
            validation_and_test_files.append(choice)
    # for very small data sets
    if math.ceil(files_in_corpus*test_percentage) > 0:
        num_test_files = math.ceil(files_in_corpus*test_percentage)
    else:
        num_test_files = 1
    test_set = validation_and_test_files[0:num_test_files]
    validation_set = [f for f in validation_and_test_files if f not in test_set]
    training_set = [f for f in files if f not in validation_and_test_files]

    return training_set, validation_set, test_set


# retrieves the labels for the files in subset. Removes file from subset, if no label is present in labels.
def lookup_label(subset, labels):
    subset_labels = {}
    for s in subset:
        l = s[:-4]
        if l in labels.keys():
            subset_labels[l] = labels[l]
        else:
            subset.remove(s)
    return subset_labels, subset


# get the labels for training, validation and test set based on the files. Removes files, for which no labels were
# found.
def get_labels(train, valid, test, path):
    with open(path, 'rb') as file:
        labels = pickle.load(file)
    train_labels, train = lookup_label(train, labels)
    valid_labels, valid = lookup_label(valid, labels)
    test_labels, test = lookup_label(test, labels)
    return train_labels, valid_labels, test_labels, train, valid, test


# load the types/genre. Notice: In the list of types "other" is not listed. ("other" is used to label the books for
# which no label was found) We add it here.
def load_types(path):
    with open(path, 'r') as file:
        types = file.readlines()
    types = [t.rstrip("\n") for t in types]
    types.append("other")
    return types


def create_data_label_tensors(data, labels, len_wordlist, generator, typelist, batchsize, path_to_data, path_to_label):
    if path.isfile(path_to_data) and path.isfile(path_to_label):
        data = torch.serialization.load(path_to_data)
        num_batches = max(int(len(data)/batchsize), 1)
        if num_batches == 1:
            batchsize = len(data)
        all_predX = torch.Tensor(num_batches, batchsize, len_wordlist)
        for i in range(num_batches):
            for j in range(batchsize):
                all_predX[i, j, :] = data[i*batchsize+j]
        labels = torch.serialization.load(path_to_label)
        all_labels_1_hot = torch.Tensor(num_batches, batchsize, len(typelist))
        for i in range(num_batches):
            for j in range(batchsize):
                all_labels_1_hot[i, j, :] = labels[i*batchsize+j]
        print("loaded data")
        print(all_predX.shape)
        print(all_labels_1_hot.shape)
    else:
        all_predX = torch.zeros(len(data), len_wordlist)
        all_labels_1_hot = torch.zeros(len(data), len(typelist))
        for i in range(len(data)):
            print(i)
            # [:-4] to remove .txt
            predX, _ = generator.generateBagOfWords(params.path_to_corpus + str(data[i]), labels[data[i][:-4]])
            labels_1_hot = torch.zeros(len(typelist))
            for j in range(len(typelist)):
                if typelist[j] in labels[data[i][:-4]]:
                    labels_1_hot[j] = 1
            all_predX[i, :] = torch.from_numpy(predX)[0].float()
            all_labels_1_hot[i, :] = labels_1_hot
        print("read data")
        torch.serialization.save(all_predX, path_to_data)
        torch.serialization.save(all_labels_1_hot, path_to_label)
        print("wrote data")
    return all_predX.to(deviceCC), all_labels_1_hot.to(deviceCC)


def predictions_correct(predictions, labels):
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    p = predictions.view(-1)
    l = labels.view(-1)
    for i in range(len(p)):
        if p[i] > 0.5:
            pred = 1
        else:
            pred = 0
        if pred == l[i]:
            if pred == 0:
                true_negative += 1
            else:
                true_positive += 1
        elif pred == 0:
            false_negative += 1
        else:
            false_positive += 1
    accuracy = (true_negative+true_positive)/(true_positive+true_negative+false_negative+false_positive)
    if true_positive == 0 and false_positive == 0:
        precision, recall = 0, 0
    else:
        precision = true_positive/(true_positive+false_positive)
        recall = true_positive/(true_positive+false_negative)
    return accuracy, precision, recall, true_positive, true_negative, false_positive, false_negative


if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # SETUP
    # -----------------------------------------------------------------------------

    random.seed(19031994)
    # create training set, validation set and test set
    training_set, validation_set, test_set = create_training_validation_test_set(params.path_to_corpus,
                                                                                 params.percentage_of_validation_data,
                                                                                 params.percentage_of_test_data)
    training_labels, validation_labels, test_labels, training_set, validation_set, test_set = \
        get_labels(training_set, validation_set, test_set, params.path_to_labels)
    wordlist = WordList()
    wordlist.loadWordList(params.path_to_word_list)
    typelist = load_types(params.path_to_genre)

    generator = FeatureGenerator(wordlist, typelist)

    all_predX_training, all_labels_training = create_data_label_tensors(training_set, training_labels, len(wordlist),
                                                                        generator, typelist, params.batchsize,
                                                                        params.training_data_tensor,
                                                                        params.training_labels_tensor)

    all_predX_validation, all_labels_validation = create_data_label_tensors(validation_set, validation_labels,
                                                                            len(wordlist), generator, typelist,
                                                                            params.batchsize,
                                                                            params.validation_data_tensor,
                                                                            params.validation_labels_tensor)

    all_predX_test, all_labels_test = create_data_label_tensors(test_set, test_labels, len(wordlist), generator,
                                                                typelist, params.batchsize, params.test_data_tensor,
                                                                params.test_labels_tensor)

    net = nn.Sequential(torch.nn.Linear(len(wordlist), 1000), #int(len(wordlist)/2)),
                        torch.nn.ReLU(),
                        #torch.nn.Linear(int(len(wordlist)/2), 1000),
                        torch.nn.Linear(1000, 100),
                        #torch.nn.ReLU(),
                        torch.nn.Linear(100, len(typelist)),
                        torch.nn.ReLU()).to(deviceCC)
    optimizer = torch.optim.Adam(net.parameters(), lr=params.learning_rate)

    # -----------------------------------------------------------------------------
    # TRAINING AND TESTING
    # -----------------------------------------------------------------------------
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', type=int, default=1, help='Distance metric for KNN')
    args = parser.parse_args()
    print(args.train)

    # train
    with open(params.path_to_output, 'a') as outfile:
        if args.train:
            print("TRAINING")
            asdasd
            outfile.write(str(time.localtime()) + "\n")
            outfile.write(str(net) + "\n")
            outfile.write("TRAINING\n")
            for epoch in range(params.num_epochs):
                if epoch%10==0:
                    print("SAVING NET")
                    torch.save(net, open("data/vanillaNN-"+str(epoch)+".pt", "wb+" ))
                epoch_tp, epoch_tn, epoch_fp, epoch_fn = 0, 0, 0, 0
                for batch in range(all_predX_training.shape[0]):
                    net.zero_grad()
                    y_pred = net(all_predX_training[batch, :, :])
                    predictions = net(all_predX_training[batch, :, :])
                    acc, pre, rec, tp, tn, fp, fn = predictions_correct(predictions, all_labels_training[batch, :, :])
                    epoch_tp += tp
                    epoch_tn += tn
                    epoch_fp += fp
                    epoch_fn += fn
                    loss = params.loss_fn(y_pred, all_labels_training[batch, :, :])
                    print(epoch, "-", batch, ":", loss.item())
                    outfile.write(str(epoch) + "-" + str(batch) + ": " + str(loss.item()) + "\n")
                    print("accuracy: ", acc, ", precision: ", pre, ", recall: ", rec)
                    outfile.write("accuracy: " + str(acc) + ", precision: " + str(pre) + ", recall: " + str(rec) +
                                  "\ntrue positive: " + str(tp) + ", true negative: " + str(tn) + ", false positive: "
                                  + str(fp) + ", false negative: " + str(fn) + "\n")
                    loss.backward()
                    optimizer.step()
                epoch_acc = (epoch_tn+epoch_tp)/(epoch_tn+epoch_tp+epoch_fn+epoch_fp)
                if epoch_tp == 0 and epoch_fp == 0:
                    epoch_pre, epoch_rec = 0, 0
                else:
                    epoch_pre = epoch_tp/(epoch_tp+epoch_fp)
                    epoch_rec = epoch_tp/(epoch_tp+epoch_fn)
                print("in epoch: accuracy: ", epoch_acc, ", precision: ", epoch_pre, ", recall: ", epoch_rec)
                outfile.write("in epoch: accuracy: " + str(epoch_acc) + ", precision: " + str(epoch_pre) + ", recall: "
                              + str(epoch_rec) + "\n")
                if epoch % 10 == 0:
                    epoch_tp, epoch_tn, epoch_fp, epoch_fn = 0, 0, 0, 0
                    for batch in range(all_predX_validation.shape[0]):
                        net.zero_grad()
                        y_pred = net(all_predX_validation[batch, :, :])
                        acc, pre, rec, tp, tn, fp, fn = predictions_correct(y_pred, all_labels_validation[batch, :, :])
                        epoch_tp += tp
                        epoch_tn += tn
                        epoch_fp += fp
                        epoch_fn += fn
                        loss = params.loss_fn(y_pred, all_labels_validation[batch, :, :])
                        print("validation loss ", batch, ":", loss.item())
                        outfile.write("validation: " + str(epoch) + "-" + str(batch) + ": " + str(loss.item()) + "\n")
                        print("accuracy: ", acc, ", precision: ", pre, ", recall: ", rec)
                        outfile.write("accuracy: " + str(acc) + ", precision: " + str(pre) + ", recall: " + str(rec) +
                                      "\ntrue positive: " + str(tp) + ", true negative: " + str(tn) + ", false positive: "
                                      + str(fp) + ", false negative: " + str(fn) + "\n")
                    epoch_acc = (epoch_tn+epoch_tp)/(epoch_tn+epoch_tp+epoch_fn+epoch_fp)
                    if epoch_tp == 0 and epoch_fp == 0:
                        epoch_pre, epoch_rec = 0, 0
                    else:
                        epoch_pre = epoch_tp / (epoch_tp + epoch_fp)
                        epoch_rec = epoch_tp / (epoch_tp + epoch_fn)
                    print("in epoch: accuracy: ", epoch_acc, ", precision: ", epoch_pre, ", recall: ", epoch_rec)
                    outfile.write("in epoch: accuracy: " + str(epoch_acc) + ", precision: " + str(epoch_pre) + ", recall: "
                                  + str(epoch_rec) + "\n")
            # test
            print("TESTING")
            outfile.write("TESTING\n")
            epoch_tp, epoch_tn, epoch_fp, epoch_fn = 0, 0, 0, 0
            for batch in range(all_predX_test.shape[0]):
                y_pred = net(all_predX_test[batch, :, :])
                acc, pre, rec, tp, tn, fp, fn = predictions_correct(y_pred, all_labels_test[batch, :, :])
                epoch_tp += tp
                epoch_tn += tn
                epoch_fp += fp
                epoch_fn += fn
                loss = params.loss_fn(y_pred, all_labels_test[batch, :, :])
                print(y_pred)
                print("test loss ", batch, ":", loss.item())
                outfile.write("test: " + str(batch) + ": " + str(loss.item()) + "\n")
                print("accuracy: ", acc, ", precision: ", pre, ", recall: ", rec)
                outfile.write("accuracy: " + str(acc) + ", precision: " + str(pre) + ", recall: " + str(rec) +
                              "\ntrue positive: " + str(tp) + ", true negative: " + str(tn) + ", false positive: "
                              + str(fp) + ", false negative: " + str(fn) + "\n")
            epoch_acc = (epoch_tn+epoch_tp)/(epoch_tn+epoch_tp+epoch_fn+epoch_fp)
            if epoch_tp == 0 and epoch_fp == 0:
                epoch_pre, epoch_rec = 0, 0
            else:
                epoch_pre = epoch_tp / (epoch_tp + epoch_fp)
                epoch_rec = epoch_tp / (epoch_tp + epoch_fn)
            print("in epoch: accuracy: ", epoch_acc, ", precision: ", epoch_pre, ", recall: ", epoch_rec)
            outfile.write("in epoch: accuracy: " + str(epoch_acc) + ", precision: " + str(epoch_pre) + ", recall: "
                          + str(epoch_rec) + "\n")

            outfile.write("\n=================================================================\n\n")
        else:
            import numpy as np
            print("GENERATING SOME DATA")
            recList = []
            accList = []
            preList = []
            lossList = []
            epochList = []
            for epoch in range(params.num_epochs):
                if epoch%10==0:
                    net = torch.load("data/vanillaNN-"+str(epoch)+".pt").to(deviceCC)
                    outfile.write("TESTING in epoch "+str(epoch)+"\n")
                    epoch_tp, epoch_tn, epoch_fp, epoch_fn = 0, 0, 0, 0
                    tempLoss = []
                    for batch in range(all_predX_test.shape[0]):
                        y_pred = net(all_predX_test[batch, :, :])
                        acc, pre, rec, tp, tn, fp, fn = predictions_correct(y_pred, all_labels_test[batch, :, :])
                        epoch_tp += tp
                        epoch_tn += tn
                        epoch_fp += fp
                        epoch_fn += fn
                        loss = params.loss_fn(y_pred, all_labels_test[batch, :, :])
                        tempLoss.append(loss.item())
                        #print(y_pred)
                        print("test loss ", batch, ":", loss.item())
                        outfile.write("test: " + str(batch) + ": " + str(loss.item()) + "\n")
                        print("accuracy: ", acc, ", precision: ", pre, ", recall: ", rec)
                        outfile.write("accuracy: " + str(acc) + ", precision: " + str(pre) + ", recall: " + str(rec) +
                                      "\ntrue positive: " + str(tp) + ", true negative: " + str(tn) + ", false positive: "
                                      + str(fp) + ", false negative: " + str(fn) + "\n")
                    epoch_acc = (epoch_tn+epoch_tp)/(epoch_tn+epoch_tp+epoch_fn+epoch_fp)
                    if epoch_tp == 0 and epoch_fp == 0:
                        epoch_pre, epoch_rec = 0, 0
                    else:
                        epoch_pre = epoch_tp / (epoch_tp + epoch_fp)
                        epoch_rec = epoch_tp / (epoch_tp + epoch_fn)
                    print("in epoch: accuracy: ", epoch_acc, ", precision: ", epoch_pre, ", recall: ", epoch_rec)
                    outfile.write("in epoch: accuracy: " + str(epoch_acc) + ", precision: " + str(epoch_pre) + ", recall: "
                                  + str(epoch_rec) + "\n")

                    outfile.write("\n=================================================================\n\n")
                    #save values
                    npArr = np.asarray(tempLoss)
                    lossList.append(np.sum(npArr)/npArr.shape[0])
                    recList.append(epoch_rec)
                    accList.append(epoch_acc)
                    preList.append(epoch_pre)
                    epochList.append(epoch)
            print(lossList)
            print(recList)
            print(accList)
            print(preList)
            print(epochList)
    print("finished")
