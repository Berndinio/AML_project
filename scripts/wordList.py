import os
import pickle
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import threading
from nltk.corpus import wordnet

from contextlib import redirect_stdout
with redirect_stdout(open(os.devnull, "w")):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

class WordList:
    def __init__(self, numThreads=16):
        self.wList = []
        self.ps = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.processedFiles = []

        self.wordSet = None
        self.threadSemaphore = threading.Semaphore(numThreads)
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.wList)

    def generateWordList(self, pPath=["data/test_corpus/", "data/train_corpus/"], restriction=[0,99999999999], lstm=False,
                         save_to="data/wordList.pkl"):
        self.wordSet = set()
        finalWordSet = set()
        threadList = []
        for path in pPath:
            for root, dirs, files in os.walk(path):
                for i,fileName in enumerate(files):
                    if(i<restriction[0] or i>restriction[1]):
                        continue
                    th = threading.Thread(target=self.generateWordListSingle, args=(fileName, path, ))
                    self.threadSemaphore.acquire()
                    th.start()
                    print("Processing "+fileName+"..."+str(i)+"/"+str(len(files)))
                    threadList.append(th)
        #wait until ALL thread have ended.
        for th in threadList:
            th.join()

        #filter unwanted things
        if lstm == True:
            self.wList.append("<PADDING>")
        for word in self.wordSet:
            filtered = self.postfilter(word)
            self.wList.append(filtered)
        self.wList = sorted(list(set(self.wList)))
        #write wordlist to file as list
        pickle.dump(self.wList, open(save_to, 'wb'))
        writeFile = open(save_to[:-4] + ".txt", 'w', encoding="utf-8")
        # writeFile = open("data/wordList.txt", 'w', encoding="utf-8")
        for word in self.wList:
            writeFile.write(word)
        return self.wList

    def generateWordListSingle(self, fileName, path):
        f = open(path+fileName, 'r', encoding="utf-8")
        self.processedFiles.append(path+fileName)
        corpus = f.read()
        #prefilter
        newCorpus = self.prefilter(corpus)

        self.lock.acquire()
        self.wordSet = self.wordSet.union(newCorpus)
        self.lock.release()
        self.threadSemaphore.release()


    def loadWordList(self, path="data/wordList.pkl"):
        pkl_file = open(path, 'rb')
        self.wList = pickle.load(pkl_file)
        return self.wList

    def index(self, word):
        return self.wList.index(word)

    def preprefilter(self, text):
        tokenized = word_tokenize(text)
        tagged = pos_tag(tokenized)
        blacklist = ["NNP"]
        #blacklist = []
        filteredList = [tag[0] for tag in tagged if not tag[1] in blacklist]
        return filteredList

    def prefilter(self, text):
        filteredList = self.preprefilter(text)
        newCorpus = set(filteredList)
        return newCorpus

    def postfilter(self, word):
        """Short summary.

        Parameters
        ----------
        word : type String
            The word which needs to be stemmed, filtered and synonymized.
        increasingWordSet : type set()
            In the beginning an empty wordset.
            Needs to be the same set() every call to ensure synonyms are filtered.

        Returns
        -------
        type
            Description of returned object.

        """
        mapping = [ '!', '"', '§','$', '%', '&', '/',
                    '(', ')', '[', ']', '{', '}', '=', '?', '´', '`',
                    '@', '€', '*', '+', '~', '#',
                    '\'', '\\', '_', ';', '<', '>', '|',
                    '.', '*', '-', ':', ',', '.', '·', '¶',
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '—']
        #filter unnecesarry not words
        #lower
        word = word.lower()
        #remove single items in set of mapping
        for charac in mapping:
            word = word.replace(charac,'')

        #filter for synonyms
        synonyms = [word]
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())

        final = self.lemmatizer.lemmatize(word)
        for syn in synonyms:
            #lemmatizing
            stemmed = self.lemmatizer.lemmatize(syn)
            if syn in self.wList:
                final = syn
                break
        return final

if __name__ == '__main__':
    WordList().generateWordList(pPath=["data/test_corpus/", "data/train_corpus/"], save_to="data/wordListAll.pkl")
