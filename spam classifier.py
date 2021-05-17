from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log, sqrt
import numpy as np
import pandas as pd
import re 

#Reading a file
trainingData_mails = pd.read_csv("TrainingData.csv")

#Pre processing a file
trainingData_mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, 
                        inplace = True)


class SpamClassifier(object):
    
    def __init__(self):
        self.spam_dict = dict()
        self.ham_dict = dict()
        self.spam_dict_prob = dict()
        self.ham_dict_prob = dict()
        self.total_spam_words = 0
        self.total_ham_words = 0
        self.total_unique_spam_words = 0
        self.total_unique_ham_words = 0
        self.spam_mails = 0
        self.ham_mails = 0
        self.spam_total_prob = 0
        self.ham_total_prob = 0
    
    #calculate TF of words in spam/ham
    def calculateTF(self):
        for i in range(trainingData_mails['v1'].size):
            label_value = trainingData_mails['v1'][i]
            words = trainingData_mails['v2'][i]
            words = words.split(' ')
            if (label_value == 'spam'):
                for i in range(len(words)):
                    self.spam_dict[words[i]] = self.spam_dict.get(words[i], 0) + 1;
                    self.total_spam_words = self.total_spam_words + 1;
                self.spam_mails = self.spam_mails + 1
            elif (label_value == 'ham'):
                for i in range(len(words)):
                    self.ham_dict[words[i]] = self.ham_dict.get(words[i], 0) + 1;
                    self.total_ham_words = self.total_ham_words + 1;
                self.ham_mails = self.ham_mails + 1
                
        self.total_unique_spam_words = len(self.spam_dict)
        self.total_unique_ham_words = len(self.ham_dict)
        
    def calculateProbForEachWord(self):
        for key in self.spam_dict:
            self.spam_dict_prob[key] = (self.spam_dict[key] + 1 ) / self.total_spam_words + self.total_unique_spam_words
        for key in self.ham_dict:
            self.ham_dict_prob[key] = (self.ham_dict[key] + 1 ) / self.total_ham_words + self.total_unique_ham_words
            
        self.spam_total_prob = self.spam_mails / trainingData_mails['v1'].size
        self.ham_total_prob = self.ham_mails / trainingData_mails['v1'].size
        
    def print(self):
       print(self.ham_total_prob)
        
        
spam_classifier = SpamClassifier()
spam_classifier.calculateTF()
spam_classifier.calculateProbForEachWord()

spam_classifier.print()