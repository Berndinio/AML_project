# several scripts for preprocessing the labels. The last one is the important one, the others are for researching the
# labels in the broadest sense

import re
from collections import Counter
import pickle

# ------------------------------------------------------------------------------------
# frequency of subjects (raw)
# ------------------------------------------------------------------------------------

raw_occurrences = 0

with open("../genre_and_preprocessing/documentwise_labels.txt", 'r') as file:
    raw = file.read()
    raw = re.sub('[0-9]*: |frozenset\(\{|\}\)|\{|\}|frozenset\(\)', '', raw)
    raw = re.sub(', [, ]+', ', ', raw)
    raw = raw[1:-3].lower()
    subjects = re.split("', '|\", \"", raw) # contains all subjects as a string

    raw_occurrences = Counter(subjects)

# with open("../genre_and_preprocessing/100_most_common_genre_raw.txt", 'w') as file:
#     file.write(str(raw_occurrences.most_common()[1:100]))

# with open("../genre_and_preprocessing/100_most_common_genre_raw.pickle", 'wb') as file:
#    pickle.dump(raw_occurrences.most_common()[1:100], file)


# ------------------------------------------------------------------------------------
# count subjects by handmade genre without regard for the documents they belong to -> may contain more than the
# documentwise version below
# ------------------------------------------------------------------------------------

genres = 0

with open("../genre_and_preprocessing/genres_bookstore.txt") as file:
    genres = file.read().splitlines()

for i in range(0, len(subjects)):
    found_genres = []#
    for gen in genres:
        if subjects[i].find(gen) > -1:
            found_genres.append(gen)
    if len(found_genres) == 0:
        subjects[i] = ["other"]
    else:
        subjects[i] = found_genres

preprocessed_occurrences = Counter([sub for subject_list in subjects for sub in subject_list])

# with open("../genre_and_preprocessing/most_common_genres_hand_chosen_overall.txt", 'w') as file:
#    file.write(str(preprocessed_occurrences))


# ------------------------------------------------------------------------------------
# count subjects by handmade genre withregard for the documents they belong to -> may contain less labels per category
# than the non-documentwise version above
# ------------------------------------------------------------------------------------

preprocessed_docwise_labels = {}
other = list()

# for searching new labels - counts the frequency of label in question. Format {"new label":0}
new_labels = {}

with open("../genre_and_preprocessing/documentwise_labels.txt", 'r') as file:
    raw = file.read()
    raw = re.sub('frozenset\(\{|\{|\}|, [0-9]+: frozenset\(\)', '', raw)
    raw = raw.lower()

    docwise_subjects = re.split("'\), |\"\), ", raw)
    raw = re.split("\}\), ", raw)

    for entry in docwise_subjects:
        id = re.match('[0-9]+:', entry).group(0)[:-1]
        label = entry[len(id)+2:]
        found_genres = []
        for gen in genres:
            if label.find(gen) > -1:
                found_genres.append(gen)
        if len(found_genres) == 0:
            preprocessed_docwise_labels[id] = ["other"]
            other.append(label)
            for lab in new_labels.keys():
                if lab in label:
                    new_labels[lab] = new_labels[lab] + 1
        else:
            preprocessed_docwise_labels[id] = found_genres

# print(other)
# print(new_labels)

frequency = Counter([l for list in preprocessed_docwise_labels.values() for l in list])

# with open("../genre_and_preprocessing/most_common_genres_hand_chosen_by_document.txt", 'w') as file:
#     file.write(str(frequency))


# ------------------------------------------------------------------------------------
# assign labels based on the "bookstore labels" per document
# ------------------------------------------------------------------------------------

preprocessed_docwise_labels = {}

with open("../genre_and_preprocessing/documentwise_labels.txt", 'r') as file:
    raw = file.read()

    # remove empty frozensets and lower entries.
    raw = re.sub('frozenset\(\{|\{|\}|, [0-9]+: frozenset\(\)', '', raw)
    raw = raw.lower()
    # entry per non-empty frozenset. Format "23479: 'subject1', 'subject2')" Instead of ' sometimes " are used.
    docwise_subjects = re.split("'\), |\"\), ", raw)

    for entry in docwise_subjects:
        id = re.match('[0-9]+:', entry).group(0)[:-1]
        label = entry[len(id)+2:]
        found_genres = []
        for gen in genres:
            if label.find(gen) > -1:
                found_genres.append(gen)
        if len(found_genres) == 0:
            preprocessed_docwise_labels[id] = ["other"]
        else:
            preprocessed_docwise_labels[id] = found_genres

with open("../genre_and_preprocessing/documentwise_labels_bookstore.txt", 'w') as file:
    file.write(str(preprocessed_docwise_labels))

with open("../genre_and_preprocessing/documentwise_labels_bookstore.pickle", 'wb') as file:
    pickle.dump(preprocessed_docwise_labels, file)
