import torch.nn as nn
import torch.optim as optim

# paths to files
path_to_word_list = "data/wordListAll.pkl"
path_to_labels = "data/genre/documentwise_labels_bookstore.pkl"
path_to_corpus = "data/train_corpus/"
path_to_genre = "data/genre/genres_bookstore.txt"

path_to_training_data_tensor = "data/lstm_training_data_tensor_el1000_bl2.pkl"
path_to_training_label_tensor = "data/lstm_training_label_tensor_el1000_bl2.pkl"
path_to_validation_data_tensor = "data/lstm_validation_data_tensor_el1000_bl2.pkl"
path_to_validation_label_tensor = "data/lstm_validation_label_tensor_el1000_bl2.pkl"
path_to_test_data_tensor = "data/lstm_test_data_tensor_el1000_bl2.pkl"
path_to_test_label_tensor = "data/lstm_test_label_tensor_el1000_bl2.pkl"

path_to_output = "data/word_based_lstm_output.log"

# percentage of validation and test data in whole data set.
batchsize = 2
excerpt_length = 1000
percentage_of_validation_data = 0.1
percentage_of_test_data = 0.03

# network parameters:
embedding_size = 20
hidden_size = 100
loss_fn = nn.MSELoss(reduction="elementwise_mean")
learning_rate = 0.0001
optimizer = optim.SGD
num_epochs = 99999
