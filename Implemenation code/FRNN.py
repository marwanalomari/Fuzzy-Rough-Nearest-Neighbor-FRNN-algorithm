# This program coded by Marwan Al Omari & Moustafa Al-Hajj
# Lebanese University, Centre for Language Sciences and Communication, Celine Centre, Tayouneh, Beirut, Lebanon
# coding: utf-8
'''This python program classifies text instances based on the lower and upper approximation of x to all the lables. This program developed based on Fuzzy Rough Nearest Neighbor (FRNN) algorithm described in the article: Jensen, R., & Cornelis, C. (2011). Fuzzy-rough nearest neighbor classification and prediction. Theoretical Computer Science, 412(42), 5871-5884
This program runnable on a command:
(python)* FRNN.py datasetname.csv
*optional
The classification is based on 5 cross_validation
classification results evaluated using the following metrics:
1.acccuracy
2.root mean squared error (RMSE)
''' 
#loading libraries
import string
import re
import sys
import math
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from random import seed
from random import randrange

#preprocessing tweets from punctuation marks, @followed by personal names, and extra spaces. transform all words to lower case
def preTweet(raw_text,split_text=False):
    #pretext = re.sub("[0-9]", "", raw_text)
    punc = "['\"./:;?,$]"
    pretext = re.sub(punc, "", raw_text)
    pretext = re.sub("\@\w+", "", pretext)
    #pretext = re.sub("\#\w+", "", pretext) 
    pretext =  pretext.strip().lower()  
    return(pretext)

#compute distance similarity between two given sentences via word2vec through vector reduction
def distanceCosine(s1,s2):
    v1 = np.mean([w2v[word] for word in preTweet(s1).split()],axis=0)
    v2 = np.mean([w2v[word] for word in preTweet(s2).split()],axis=0)
    num = 0
    den1 = 0
    den2 = 0
   
    for i in range(len(v1)):
        num = num + v1[i]*v2[i]
        den1 = den1 + v1[i]*v1[i]
        den2 = den2 + v2[i]*v2[i]
    res = num/(math.sqrt(den1)*math.sqrt(den2))
    return res

#compute Euclidean distance for similarity relations via word2vec through vector reduction
def distanceEuclidean(s1,s2):
    v1 = np.mean([w2v[word] for word in preTweet(s1).split()],axis=0)
    v2 = np.mean([w2v[word] for word in preTweet(s2).split()],axis=0)
    num = 0
   
    for i in range(len(v1)):
        num = num + (v1[i]-v2[i])*(v1[i]-v2[i])
    res = math.sqrt(num)
    return res

#getting nearest neighbours to y_objects by using Euclidean similarity, and sorting the dictionary based on the similarity values (descending order). Then, appending texts with corresponding labels
def getNearestNeighbours (X,Y,y, K):
    arr = dict();
    for t in X:
       arr[t] = distanceEuclidean(t,y)
    
    arr1 = dict()
    arr1 = sorted([(v, k) for (k, v) in arr.items()],reverse=True)
    
    res = []
    u=""
    for val, tweet in arr1[:K]:
        #print (val)
        u=tweet+":"+str(Y[X.index(tweet)])
        res.append(u)
    return res

#cleaning y_labels from texts, leaving only numbers
def cleanText(raw_text):
    numbers_only = raw_text.split(":")
    return(numbers_only[0])

# R of attribute a computation
'''Ra=1-(|a(y)-a(z)/amax-amin|)'''
def Ra(a,y,z,X):
    
    #Calculate the max and min of values of a specific attribute of all elements of the training set
    vx = 0
    va = list()
    for x in X:
        vx = np.mean([w2v[word] for word in preTweet(x).split()],axis=0)
        va.append(vx[a])
    amax = max(va)
    amin = min(va)
    ###########
    
    v = 0
    #v1 is the embedding vector of y
    v1 = np.mean([w2v[word] for word in preTweet(y).split()],axis=0)
    #v2 is the embedding vector of z
    v2 = np.mean([w2v[word] for word in preTweet(z).split()],axis=0)
    v = (1-abs(v1[a]-v2[a]))/abs(amax-amin)
    return v

#R computation
'''R=min(Ra(y,z))'''
def R(y,z,X):
    values = list()
    for a in range(64):
        values.append(Ra(a,y,z,X))
    return min(values)
        
#lower approximation computation
'''(R l c) (y)= Inf l((R(y,z),C(z)))'''
def lower(X,Y,y,C):
    
    values = list()
    v = 0
    c=0
    d=0
    #p = 0
    for z in X:
        #p+=1
        #print ("Lower",p)
        c = 1-R(y,z,X)
        if(Y[X.index(z)]==C):
            d=1
        else:
            d=0
        if(c>d):
            v= c
        else:
            v = d
        values.append(v)
    return max(values)

#upper approximation computation
'''(R u c) (y)= sup t((R(y,z),C(z)))'''
def upper(X,Y,y,C):
    values = list()
    v = 0
    c=0
    d=0
    #p = 0
    for z in X:
        #p+=1
        #print ("Upper",p)
        c = R(y,z,X)
        if(Y[X.index(z)]==C):
            d=1
        else:
            d=0
        if(c<d):
            v= c
        else:
            v = d
        values.append(v)
    return min(values)

#The fuzzy-rough nearest neighbour algorithm (FRNN)
def FRNN(X,Y,y,K):
    N = list()
    N = getNearestNeighbours(X,Y,y,K)
    Classes = []
    Class = 10
    for n in N:
        Classes.append(int(n.split(":")[1]))
    T = 0
    for C in Classes:
        if((lower(X,Y,y,C)+upper(X,Y,y,C))/2>=T):
            Class = C
            T = (lower(X,Y,y,C)+upper(X,Y,y,C))/2
        
    return Class
    

#Loading dataset
script = sys.argv[0]
filename = sys.argv[1]
df = pd.read_csv(filename, encoding="utf-8")
#spliting dataset to text data & labels
X_train=df.Tweet
y_train=df['Intensity Class']

X_train_cleaned = []
y_train_cleaned = []

#using pretweet function to clean x_training_data
for d in X_train:
    X_train_cleaned.append(preTweet(d))
#using cleanText function to clean y_training_data
for d in y_train:
    y_train_cleaned.append(int(cleanText(d)))

#prepring tweets for word2vec parsing
def parseSent(review):
    raw_sentences = review.split()
    sentences = []
    sentences.append(raw_sentences)
    return sentences

sentences = []
for review in X_train_cleaned:
    sentences += parseSent(review)

#training word2vec model with the following parameters:
num_features = 64 #dimensions                  
min_word_count = 1 #minimum word occurrence     
num_workers = -1 #processors      
context = 5 #context words
###########
w2v = Word2Vec(sentences, workers=num_workers, size=num_features, 
min_count = min_word_count, window = context,sg = 0)
w2v.init_sims(replace=True)
#saving the model to the desk drive
w2v.save("word2vec_model")

#model = Word2Vec.load("word2vec_model")

# getNearestNeighbours(X_train_cleaned,y_train_cleaned,'the dreams',10)
#print ("The corresponding class is ",FRNN(X_train_cleaned,y_train_cleaned,'the dreams',10))

#accuracy metric for assesing the algorithm
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.

# Calculate root mean squared error from actual & predicated labels (stdv)
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return math.sqrt(mean_error)

# Split a dataset into k folds through indexes
def cross_validation_indexes(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(index)
        dataset_split.append(fold)
    return dataset_split

#remove index (id_to_del) from y_train int list (l)
def deletioni(l, id_to_del):
    arr = np.array(l, dtype='int')
    return list(np.delete(arr, id_to_del))

#remove index (id_to_del) from x_train str list (l)
def deletions(l, id_to_del):
    arr = np.array(l, dtype='str')
    return list(np.delete(arr, id_to_del))
indexes = cross_validation_indexes(X_train_cleaned,5)

####test area####
#print (lower(X_train_cleaned,y_train_cleaned,'the dreams',2))
#print (upper(X_train_cleaned,y_train_cleaned,'the dreams'))
####end####

#main operation
#5 continous loop
for i in range(5):
    #print ("round completed")
    output_class=0
    predicated=list()
    actual=list()
    totalaccuracy=list()
    acc=0
    A=0
    std=0
    for x in indexes[i]:
        #print(X_train_cleaned[x])
        #remove x fold index from training & testing sets
        x_train = deletions(X_train_cleaned,x)
        y_train = deletioni(y_train_cleaned,x)
        #print(y_train)
        #print(x_train)
        #output class for [x] test fold i
        output_class = FRNN(x_train,y_train,X_train_cleaned[x],5)
        #push predicated & actual labels
        predicated.append(output_class)
        actual.append(y_train_cleaned[x])
        if len(actual)==len(indexes[i]):
            #calculating accuracy
            A = accuracy_metric(actual,predicated)
            #calculating variance RMSE
            std = rmse_metric(actual,predicated)
            print("Mean classification accuracy for fold "+str(i+1)+" is "+str(A)+", standard deviation "+str(std))

'''end of the program'''