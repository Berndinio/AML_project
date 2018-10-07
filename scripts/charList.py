import os
import pickle
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import threading


class CharList:
    def __init__(self):
        self.charString = """!\"§$%&/()=?\\`´@*'_:;,.-#+<>|^°
                        qwertzuiopasdfghjklyxcvbnm
                        QWERTZUIOPASDFGHJKLYXCVBNM"""

    def __len__(self):
        return len(self.charString)

    def index(self, pChar):
        #returns -1 if not in charString
        return self.charString.find(pChar)
