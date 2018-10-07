import torch.nn as nn

# paths to files
path_to_labels = "data/genre/documentwise_labels_bookstore.pkl"
path_to_genre = "data/genre/genres_bookstore.txt"

# 40 test documents
path_to_word_list = "data/wordListAll.pkl"
path_to_corpus = "data/train_corpus/"

training_data_tensor = "data/vanilla_training_data_tensor40.pkl"
training_labels_tensor = "data/vanilla_training_labels_tensor40.pkl"
validation_data_tensor = "data/vanilla_validation_data_tensor40.pkl"
validation_labels_tensor = "data/vanilla_validation_labels_tensor40.pkl"
test_data_tensor = "data/vanilla_test_data_tensor40.pkl"
test_labels_tensor = "data/vanilla_test_labels_tensor40.pkl"

path_to_output = "data/vanilla_NN_output.log"

# 20 test documents
# path_to_word_list = "data/katja/wordList20.pkl"
# path_to_corpus = "data/test_corpus/"
#
# training_data_tensor = "data/katja/training_data_tensor20.pkl"
# training_labels_tensor = "data/katja/training_labels_tensor20.pkl"
# validation_data_tensor = "data/katja/validation_data_tensor20.pkl"
# validation_labels_tensor = "data/katja/validation_labels_tensor20.pkl"
# test_data_tensor = "data/katja/test_data_tensor20.pkl"
# test_labels_tensor = "data/katja/test_labels_tensor20.pkl"

# percentage of validation and test data in whole data set.
percentage_of_validation_data = 0.1
percentage_of_test_data = 0.01

num_epochs = 100
batchsize = 16
learning_rate = 0.0001
loss_fn = nn.SmoothL1Loss()
