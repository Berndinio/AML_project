import math
import numpy as np
from os import path, walk
import pickle as pkl
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# local imports
import scripts.wordbased_lstm_params as params
import scripts.featureGenerator as gen
import scripts.wordList as wl

if torch.cuda.is_available():
    deviceCC = torch.device('cuda:0')
else:
    deviceCC = torch.device('cpu')

# load the types/genre. Notice: In the list of types "other" is not listed. ("other" is used to label the books for
# which no label was found) We add it here.
def load_types(path):
    with open(path, 'r') as file:
        types = file.readlines()
    types = [t.rstrip("\n") for t in types]
    types.append("other")
    return types


def one_hot_label(typelist, doc_labels):
    one_hot = np.zeros(len(typelist))
    for i in range(len(typelist)):
        if typelist[i] in doc_labels:
            one_hot[i] = 1
    return one_hot


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


def create_input_tensors(gen, len_wordlist, typelist, batchsize, input_set, path_corpus, path_labels, path_data_tensor,
                         path_label_tensor):
    if path.isfile(path_data_tensor) and path.isfile(path_label_tensor):
        data_tensor = pkl.load(open(path_data_tensor, 'rb'))
        label_tensor = pkl.load(open(path_label_tensor, 'rb'))
    else:
        labels = pkl.load(open(params.path_to_labels, 'rb'))
        print("loaded files")
        # create tensor
        num_batches = max(int(len(input_set)/batchsize), 1)
        data_tensor = np.zeros([num_batches, batchsize, params.excerpt_length])
        label_tensor = np.zeros([num_batches, batchsize, len(typelist)])

        print("creating embedding - this may take a while, especially for larger files")
        i = 0
        j = 0
        for file in input_set:
            if i >= num_batches*batchsize:
                break
            print(i, ", apprx. length: ", len(open(path_corpus+file, 'r').read().split(" ")))
            predX, _ = gen.generate_LSTM_indexed(path_corpus+file, typelist)
            print("document shape: ", predX.shape)
            if predX.shape[1] > params.excerpt_length:
                random_start = np.random.randint(0, predX.shape[1]-params.excerpt_length)
                data_tensor[int(j/batchsize), j%batchsize, :] = predX[0][random_start:random_start+params.excerpt_length]
                label_tensor[int(j/batchsize), j%batchsize, :] = one_hot_label(typelist, labels[file[:-4]])
                j += 1
            i += 1

        # remove 0-entries
        if len(input_set) > batchsize:
            data_tensor = data_tensor[0:int(j/batchsize), :, :]
            label_tensor = label_tensor[0:int(j/batchsize), :, :]
        else:
            data_tensor = data_tensor[:, 0:j, :]
            label_tensor = label_tensor[:, 0:j, :]

        print("data: ", data_tensor)
        print("labels: ", label_tensor)

        print("extracted files, creating tensors")
        # convert to torch.Tensor
        data_tensor = torch.as_tensor(data_tensor).long()
        label_tensor = torch.as_tensor(label_tensor).float()
        pkl.dump(data_tensor, open(path_data_tensor, 'wb'))
        pkl.dump(label_tensor, open(path_label_tensor, 'wb'))
    return data_tensor.to(deviceCC), label_tensor.to(deviceCC)


# the implementation is based on the tutorial at
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
class LSTMModell(nn.Module):
    def __init__(self, num_batches, embedding_dim, hidden_dim, num_labels, num_words):
        super(LSTMModell, self).__init__()
        self.num_batches = num_batches
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        self.embedding = nn.Embedding(num_words, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_to_labels = nn.Linear(hidden_dim, num_labels)

        self.hidden = self.init_hidden(1)

    def init_hidden(self, inputsize):
        return (torch.zeros(1,inputsize,self.hidden_dim), torch.zeros(1,inputsize,self.hidden_dim))

    def forward(self, input_sequence):
        embeds = self.embedding(input_sequence)
        lstm_out, self.hidden = self.lstm(embeds.view(input_sequence.shape[1], input_sequence.shape[0], -1),
                                          self.hidden)
        label_space_wordwise = self.hidden_to_labels(lstm_out)
        s = nn.Sigmoid()
        # Using log-space to prevent underflows. We assume (due to lack of time), that the probabilities per predicted
        # class are the same.
        label_space_docwise = torch.sum(torch.log(s(label_space_wordwise)), dim=0)
        label_space_docwise += torch.log(torch.full(label_space_docwise.shape, 1/self.num_labels))
        output = F.softmax(label_space_docwise, dim=1)
        return output


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
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negative)
    return accuracy, precision, recall, true_positive, true_negative, false_positive, false_negative

    # max in prediction
    # if max == 1 in label: true, else false


if __name__ == '__main__':
    random.seed(19031994)
    torch.manual_seed(19031994)
    print("---------------- SETUP DATA AND LABELS ----------------")
    # create input vector
    wordList = wl.WordList()
    if path.isfile(params.path_to_word_list):
        wordList.loadWordList(params.path_to_word_list)
    else:
        wordList.generateWordList(lstm=True, save_to=params.path_to_word_list)
    typelist = load_types(params.path_to_genre)
    generator = gen.FeatureGenerator(wordList, typelist)

    # create training, validation and test set
    training_set, validation_set, test_set = create_training_validation_test_set(params.path_to_corpus,
                                                                                 params.percentage_of_validation_data,
                                                                                 params.percentage_of_test_data)

    print(training_set, validation_set, test_set)

    training_data_tensor, training_label_tensor = create_input_tensors(generator, len(wordList), typelist,
                                                                       params.batchsize, training_set,
                                                                       params.path_to_corpus, params.path_to_labels,
                                                                       params.path_to_training_data_tensor,
                                                                       params.path_to_training_label_tensor)

    validation_data_tensor, validation_label_tensor = create_input_tensors(generator, len(wordList),
                                                                           typelist, params.batchsize, validation_set,
                                                                           params.path_to_corpus, params.path_to_labels,
                                                                           params.path_to_validation_data_tensor,
                                                                           params.path_to_validation_label_tensor)

    test_data_tensor, test_label_tensor = create_input_tensors(generator, len(wordList), typelist,
                                                               params.batchsize, test_set, params.path_to_corpus,
                                                               params.path_to_labels, params.path_to_test_data_tensor,
                                                               params.path_to_test_label_tensor)

    print(training_data_tensor.shape, validation_data_tensor.shape, test_data_tensor.shape)

    print("---------------- SETUP NETWORK ----------------")
    input_dim = params.excerpt_length
    net = LSTMModell(1, params.embedding_size, params.hidden_size, len(typelist), len(wordList)).to(deviceCC)

    print("NN:")
    print(net)

    print("---------------- TRAIN AND TEST NETWORK ----------------")

    loss_function = params.loss_fn
    optimizer = params.optimizer(net.parameters(), params.learning_rate)

    with open(params.path_to_output, 'a') as out_file:
        out_file.write(str(time.localtime()) + "\n")
        out_file.write("TRAINING" + "\n")
        print("TRAINING")
        for epoch in range(params.num_epochs):
            if epoch%10==0:
                print("SAVING NET")
                torch.save(net, open("data/vanillaNN-"+str(epoch)+".pt", "wb+" ))
            epoch_tp, epoch_tn, epoch_fp, epoch_fn = 0, 0, 0, 0
            for batch in range(training_data_tensor.shape[0]):
                net.zero_grad()
                net.hidden = net.init_hidden(training_data_tensor.shape[1])
                predictions = net(training_data_tensor[batch, :, :])
                acc, pre, rec, tp, tn, fp, fn = predictions_correct(predictions, training_label_tensor[batch, :, :])
                epoch_tp += tp
                epoch_tn += tn
                epoch_fp += fp
                epoch_fn += fn
                loss = params.loss_fn(predictions, training_label_tensor[batch, :, :])
                print(epoch, "-", batch, ": ", loss.item())
                out_file.write(str(epoch) + "-" + str(batch) + ": " + str(loss.item()) + "\n")
                print("accuracy: ", acc, ", precision: ", pre, ", recall: ", rec)
                out_file.write("accuracy: " + str(acc) + ", precision: " + str(pre) + ", recall: " + str(rec) +
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
            out_file.write("in epoch: accuracy: " + str(epoch_acc) + ", precision: " + str(epoch_pre) + ", recall: "
                           + str(epoch_rec) + "\n")
            if (epoch % 5) == 0:
                epoch_tp, epoch_tn, epoch_fp, epoch_fn = 0, 0, 0, 0
                for batch in range(validation_data_tensor.shape[0]):
                    with torch.no_grad():
                        net.hidden = net.init_hidden(validation_data_tensor.shape[0])
                        predictions = net(validation_data_tensor[batch, :, :])
                        acc, pre, rec, tp, tn, fp, fn = predictions_correct(predictions,
                                                                            validation_label_tensor[batch, :, :])
                        epoch_tp += tp
                        epoch_tn += tn
                        epoch_fp += fp
                        epoch_fn += fn
                        loss = params.loss_fn(predictions, validation_label_tensor[batch, :, :])
                        print("validation: ", epoch, "-", batch, ": ", loss.item())
                        out_file.write("validation: " + str(epoch) + "-" + str(batch) + ": " + str(loss.item()) + "\n")
                        print("accuracy: ", acc, ", precision: ", pre, ", recall: ", rec)
                        out_file.write("accuracy: " + str(acc) + ", precision: " + str(pre) + ", recall: " + str(rec) +
                                       "\ntrue positive: " + str(tp) + ", true negative: " + str(tn) +
                                       ", false positive: " + str(fp) + ", false negative: " + str(fn) + "\n")
                epoch_acc = (epoch_tn+epoch_tp)/(epoch_tn+epoch_tp+epoch_fn+epoch_fp)
                if epoch_tp == 0 and epoch_fp == 0:
                    epoch_pre, epoch_rec = 0, 0
                else:
                    epoch_pre = epoch_tp / (epoch_tp + epoch_fp)
                    epoch_rec = epoch_tp / (epoch_tp + epoch_fn)
                print("in epoch: accuracy: ", epoch_acc, ", precision: ", epoch_pre, ", recall: ", epoch_rec)
                out_file.write("in epoch: accuracy: " + str(epoch_acc) + ", precision: " + str(epoch_pre) + ", recall: "
                               + str(epoch_rec) + "\n")
        print("TESTING")
        out_file.write("TESTING\n")
        epoch_tp, epoch_tn, epoch_fp, epoch_fn = 0, 0, 0, 0
        for batch in range(test_data_tensor.shape[0]):
                with torch.no_grad():
                    net.hidden = net.init_hidden(test_data_tensor.shape[0])
                    predictions = net(test_data_tensor[batch, :, :])
                    acc, pre, rec, tp, tn, fp, fn = predictions_correct(predictions, test_label_tensor[batch, :, :])
                    epoch_tp += tp
                    epoch_tn += tn
                    epoch_fp += fp
                    epoch_fn += fn
                    loss = params.loss_fn(predictions, test_label_tensor[batch, :, :])
                    print(predictions)
                    print("test: ", batch, ": ", loss.item())
                    out_file.write("test: " + str(batch) + ": " + str(loss.item()) + "\n")
                    print("accuracy: ", acc, ", precision: ", pre, ", recall: ", rec)
                    out_file.write("accuracy: " + str(acc) + ", precision: " + str(pre) + ", recall: " + str(rec) +
                                   "\ntrue positive: " + str(tp) + ", true negative: " + str(tn) + ", false positive: "
                                   + str(fp) + ", false negative: " + str(fn) + "\n")
        epoch_acc = (epoch_tn+epoch_tp)/(epoch_tn+epoch_tp+epoch_fn+epoch_fp)
        if epoch_tp == 0 and epoch_fp == 0:
            epoch_pre, epoch_rec = 0, 0
        else:
            epoch_pre = epoch_tp / (epoch_tp + epoch_fp)
            epoch_rec = epoch_tp / (epoch_tp + epoch_fn)
        print("in epoch: accuracy: ", epoch_acc, ", precision: ", epoch_pre, ", recall: ", epoch_rec)
        out_file.write("in epoch: accuracy: " + str(epoch_acc) + ", precision: " + str(epoch_pre) + ", recall: "
                       + str(epoch_rec) + "\n")

        out_file.write("\n=================================================================\n\n")

    print("finished")
