from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import string

#Reading a file
trainingData_mails = pd.read_csv('C:/Users/dchandra/Downloads/Mine/Study/Projects/spam_classifier/TrainingData.csv')
testData_mails = pd.read_csv('C:/Users/dchandra/Downloads/Mine/Study/Projects/spam_classifier/TestData.csv')

trainingData_mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, 
                        inplace = True)
testData_mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, 
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
        self.total_unique_words = 0
        self.spam_mails = 0
        self.ham_mails = 0
        self.spam_total_prob = 0
        self.ham_total_prob = 0
        
    def preprocess(self, sentence):
        sentence = sentence.lower()
        #gram = 2
        words = word_tokenize(sentence)
        #if gram > 1:
        #    w = []
        #    for i in range(len(words) - gram + 1):
        #        w += [' '.join(words[i:i + gram])]
        #    return w
        words = [w for w in words if len(w) > 2]
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
        words = [word.strip(string.punctuation) for word in words if word.strip(string.punctuation) != '']
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
        return words
    
    #calculate TF of words in spam/ham
    def calculateTF(self):
        processed_words = dict()
        for i in range(trainingData_mails['v1'].size):
            label_value = trainingData_mails['v1'][i]
            message = trainingData_mails['v2'][i]
            words = self.preprocess(message)
            if (label_value == 'spam'):
                for i in range(len(words)):
                    self.spam_dict[words[i]] = self.spam_dict.get(words[i], 0) + 1;
                    self.total_spam_words = self.total_spam_words + 1;
                    if(words[i] not in processed_words):
                        self.total_unique_words += 1
                        processed_words[words[i]] = 1
                self.spam_mails = self.spam_mails + 1
            elif (label_value == 'ham'):
                for i in range(len(words)):
                    self.ham_dict[words[i]] = self.ham_dict.get(words[i], 0) + 1;
                    self.total_ham_words = self.total_ham_words + 1;
                    if(words[i] not in processed_words):
                        self.total_unique_words += 1
                        processed_words[words[i]] = 1
                self.ham_mails = self.ham_mails + 1
                
        self.total_unique_spam_words = len(self.spam_dict)
        self.total_unique_ham_words = len(self.ham_dict)
        
    def calculateProbForEachWord(self):
        for key in self.spam_dict:
            self.spam_dict_prob[key] = (self.spam_dict[key]) / (self.total_spam_words)
        for key in self.ham_dict:
            self.ham_dict_prob[key] = (self.ham_dict[key]) / (self.total_ham_words)
            
        self.spam_total_prob = self.spam_mails / trainingData_mails['v1'].size
        self.ham_total_prob = self.ham_mails / trainingData_mails['v1'].size
        
    def predict(self, testData):
        result = list()
        for body in testData:
           result.append(self._predict(body)[0])
        return result
           
    def _predict(self, body):
        spam_score = 0
        ham_score = 0
        words = self.preprocess(body)
        for word in words:
            if (word in self.spam_dict_prob):
                spam_score += self.spam_dict_prob[word]
            else:
                spam_score += 1 / (self.total_spam_words + self.total_unique_words)
            if (word in self.ham_dict_prob):
                ham_score += self.ham_dict_prob[word]
            else:
                ham_score += 1 / (self.total_ham_words + self.total_unique_words)
        if(spam_score > ham_score):
            return 'spam', spam_score
        elif(spam_score < ham_score):
            return 'ham', ham_score
        else:
            return 'Not able to decide'
    
    def accuracy(self, result, value):
        crct = 0
        for i in range(len(result)):
            if(value == result[i]):
                crct += 1
        print(crct)
        print(len(result))
        print(crct/len(result))
           
    def accuracyoverall(self, result):
       crct = 0
       for i in range(len(result)):
           if(testData_mails['v1'][i] == result[i]):
               crct += 1
       print(crct)
       print(len(result))
       print(crct/len(result))
        
spam_classifier = SpamClassifier()
spam_classifier.calculateTF()
spam_classifier.calculateProbForEachWord()

print('****************accuracy for spam****************')
test = list()
for i in range(len(testData_mails)):
    if(testData_mails['v1'][i] == 'spam'):
        test.append(testData_mails['v2'][i])
result = spam_classifier.predict(test)
spam_classifier.accuracy(result, 'spam')

print('****************accuracy for ham****************')
test = list()
for i in range(len(testData_mails)):
    if(testData_mails['v1'][i] == 'ham'):
        test.append(testData_mails['v2'][i])
result = spam_classifier.predict(test)
spam_classifier.accuracy(result, 'ham')

print('****************Overrall accuracy***************')

test = list()
for i in range(len(testData_mails)):
    test.append(testData_mails['v2'][i])
result = spam_classifier.predict(test)
spam_classifier.accuracyoverall(result)
print('********************************')

print(spam_classifier._predict("Did you show him and wot did he say or could u not c him 4 dust?")) #ham
print(spam_classifier._predict("Not heard from U4 a while. Call 4 rude chat private line 01223585334 to cum. Wan 2C pics of me gettin shagged then text PIX to 8552. 2End send STOP 8552 SAM xxx")) #spam
#print(trainingData_mails['v1'].value_counts())
#spam_messages = trainingData_mails[trainingData_mails['v1'] == 'spam']
