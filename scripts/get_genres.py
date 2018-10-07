# get a random sample of labels.

from gutenberg.query import get_metadata
import random

random.seed(a=19031003)
random_docs = random.sample(range(57700), 100)
labels = set()

for doc in random_docs:
    subjects = get_metadata('subject', doc)
    for s in subjects:
        labels.add(s)

with open('genres.txt', 'w') as file:
    file.write(str(labels))