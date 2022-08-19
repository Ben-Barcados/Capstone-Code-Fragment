#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import pandas as pd
import numpy as np
import math
import itertools
from matplotlib import pyplot as plt
import librosa.display
import sklearn
import os
import random
import pathlib
import csv
from matplotlib import cm as cm
from sklearn.metrics import confusion_matrix  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
#used to create the confusion matrix
from sklearn.metrics import confusion_matrix
import dataframe_image as dfi
from scipy.io.wavfile import write
import IPython.display as ipd
from sklearn.linear_model import LinearRegression
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Collect wav file
voice_data = 'nba_podcast.wav'
v , sr = librosa.load(voice_data)



# In[3]:


sr


# In[4]:


v.shape


# In[5]:


v2 = v[22050:44100]
#Visualizing Audio Array
plt.figure(figsize=(14, 5))
librosa.display.waveplot(v2, sr=sr)


# In[6]:


v_sz,  = v.shape
print(v_sz)


# In[9]:


numS = (int)(v_sz/sr)
print(numS)
mfcc = librosa.feature.mfcc(y=v, sr=sr)


# In[23]:


for x in range(numS):
    print(x)


# In[7]:


#white noise
RMS=math.sqrt(np.mean(v**2))
print(RMS)
wnl = noise=np.random.normal(0, (RMS/50), v.shape[0])
#Add white noise
vfw = v + wnl
#audio with white noise
ipd.Audio(data = vfw, rate = sr)


# In[ ]:


for x in range(100):
    
    startI = x*sr
    if(((x+1)*sr)>vfw.shape[0]):
        endI = vfw.shape[0] - (sr*x)
    else:
        endI = (x+1)*sr
    #print(x)
    cutP1 = random.randint(startI,(endI-500))
    cutP2 = random.randint(cutP1,endI)
    vtemp = vfw
    vtemp[cutP1:cutP2] = wnl[cutP1:cutP2]
    y = vtemp[startI:endI]
    #plt.title("Signal Cut Time Domain")
    #plt.figure(figsize=(14, 5))
    #librosa.display.waveplot(y, sr=sr)
    # Zooming in
    n0 = cutP1-500-startI
    n1 = cutP2+500-endI
    plt.title("Signal Cut Time Domain Zoomed")
    plt.ylabel("Amplitdue")
    plt.xlabel("Frames")
    plt.figure(figsize=(14, 5))
    plt.plot(y[n0:n1])
    plt.grid()
    


# In[100]:


#Create dataset headers
header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' cut'
header = header.split()

#Create file for dataset
file = open('nba_dataset.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
    
##Add data to dataset    
for x in range(numS):
    
    startI = x*sr
    if(((x+1)*sr)>vfw.shape[0]):
        endI = vfw.shape[0] - (sr*x)
    else:
        endI = (x+1)*sr
    #print(x)
    y=vfw[startI:endI]
    rmse = librosa.feature.rms(y=y)[0]
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    to_append += f' {0}'
    file = open('nba_dataset.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
        
    if((x%3)==0):
        cutP1 = random.randint(startI,(endI-500))
        cutP2 = random.randint(cutP1,endI)
        vtemp = vfw
        vtemp[cutP1:cutP2] = wnl[cutP1:cutP2]
        y = vtemp[startI:endI]
        rmse = librosa.feature.rms(y=y)[0]
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {1}'
        file = open('nba_dataset.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
        


# In[102]:


data = pd.read_csv('nba_dataset.csv')
data.head()# Dropping unneccesary columns
target_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(target_list)#Scaling the Feature columns
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[103]:


# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 

    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=5) 

    # Performing training 
    clf_gini.fit(X_train, y_train)
    
    return clf_gini

# Function to perform training with entropy. 
def train_using_entropy(X_train, X_test, y_train): 

    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth = 5, min_samples_leaf = 5) 

    # Performing training 
    clf_entropy.fit(X_train, y_train)
    
    return clf_entropy

#Function preforms training with random forest analysis
def train_using_rfc(X_train, X_test, y_train):
    
    #Random Forest Classifier object
    rfc = RandomForestClassifier(n_estimators=200)
    
    #Preform Training
    rfc.fit(X_train, y_train)
    
    
    return rfc

#Function preforms training with stochastic gradient decent classifier 
def train_using_sgd(X_train, X_test, y_train):
    
    #Stochastic Gradient Decent Classifier
    sgd = SGDClassifier(penalty=None)
    
    #Preform Training
    sgd.fit(X_train, y_train)
    
    return sgd

#Functions preforms training with support vector classifier
def train_using_svc(X_train, X_test, y_train):
    
    #Support Vector Classifier
    svc = SVC()
    
    #Preform Training
    svc.fit(X_train, y_train)
    
    
    
    return svc


# In[104]:


# Function to make predictions 
def prediction(X_test, clf_object): 

    # Predicton 
    y_pred = clf_object.predict(X_test) 
    #print("Predicted values:") 
    #print(y_pred) 
    return y_pred 

# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, title, cmap=plt.cm.Blues): 

    print ("Accuracy : \n", 
    accuracy_score(y_test,y_pred)*100) 
    
    #print("Confusion Matrix: \n")
    cm = confusion_matrix(y_test, y_pred)
    #full_cm = add_dims(cm,2,2)
    
    
    #pp = PdfPages('Wine_Confusion_Matricies.pdf')#creates pdf to save figures
    plot_conf_matrix(y_test, y_pred, cm, np.unique(y_test), True, title)
    #plt.savefig(pp, format='pdf')#appends the figure to the pdf
    #plot_conf_matrix(y_test, y_pred, cm, np.array([1,2,3,4,5,6,7,8,9,10]), True, "Full Confusion Matrix")
    #plt.savefig(pp, format='pdf')#appends the figure to the pdf
    #pp.close()#closes the pdf and saves it
    return accuracy_score(y_test, y_pred)*100
    
#Takes in the confusion matrix and plots it
def plot_conf_matrix(y_test, y_pred, cm, classes, in_percent=True, title='Confusion Matrix', cmap=plt.cm.Blues):
    #takes in the confusion matrix (cm), the class names, if the values should be reported in %, the title, and the color
    
    
    #Sets up the figure with title, tickmarks, axis labels,and a colorbar legend
    plt.figure()
    plt.imshow(cm,interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))#sets up tick marks with labels
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    #uses threshold for determining text color
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):#creates plot with text in it
        plt.text(j, i, cm[i,j], horizontalalignment = "center", color="white" if cm[i,j] > thresh else "black")

    


#cm = confusion_matrix(y_test, y_pred)
#full_cm = add_dims(cm,2,2)


# In[105]:


#Training Trees
clf_gini = train_using_gini(X_train, X_test, y_train) 
clf_entropy = train_using_entropy(X_train, X_test, y_train) 
rfc = train_using_rfc(X_train, X_test, y_train)
sgd = train_using_sgd(X_train, X_test, y_train)
svc = train_using_svc(X_train, X_test, y_train)

# Prediction using Gini 
print ("Results Using Gini Index:") 
y_pred_gini = prediction(X_test, clf_gini) 
Agini = cal_accuracy(y_test, y_pred_gini, "Gini Confusion Matrix") 
    
# Prediction using Entropy
print ("Results Using Entropy:")  
y_pred_entropy = prediction(X_test, clf_entropy) 
Aent = cal_accuracy(y_test, y_pred_entropy, "Entropy Confusion Matrix") 
    
    
# Prediction using Random Forest Classifier
print ("Results Using Random Forest Classifier:")  
y_pred_rfc = prediction(X_test, rfc) 
Arfc=cal_accuracy(y_test, y_pred_rfc, "RFC Confusion Matrix")

    
# Prediction using Stochastic Gradient Decent Classifier
print ("Results Using Stochastic Gradient Decent Classifier:")  
y_pred_sgd = prediction(X_test, sgd) 
Asgd=cal_accuracy(y_test, y_pred_sgd, "SGD Confusion Matrix")
    
# Prediction using Support Vector Classifier
print ("Results Using Support Vector Classifier:") 
y_pred_svc = prediction(X_test, svc) 
Asvc=cal_accuracy(y_test, y_pred_svc, "SVC Confusion Matrix")


# In[106]:


from keras import layers
from keras import layers
import keras
from keras.models import Sequential


# In[153]:


model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[154]:


classifier = model.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=128)


# In[155]:


score = model.evaluate(X_test, y_test, verbose = 0)


# In[156]:


score[1]


# In[158]:


from pickle import dump


# In[159]:


dump(sgd, open('sgd_model.pkl', 'wb'))

dump(encoder, open('sgd_encoder.pkl', 'wb'))
dump(scaler, open('sgd_scaler.pkl', 'wb'))

