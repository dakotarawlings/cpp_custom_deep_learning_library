"""
Script to build a model for classifying images of handwritten digits

This script utilizes Dakota's feed forward neural network model building library written in c++

The script builds and trains a FFNN model on the MNIST database of handwritten charachters and saves/serializes the model as a .p file

"""
#import data processing tools
import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#import data visualization tools
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


#import custom DLL for building FFNN models 
try:
    import FFNN_pymodule
except:
    pass

#import pickle to save the model
import pickle

#Import the MNIST database fro charachter recognition (row has  aflattened 784 feature array of pixels for a given charachter)
data_path=r"training_data/MNIST_train.csv"
full_df=pd.read_csv(data_path)

#To make the trianing faster we can optionally take a subset of the data to expadite training
df=full_df.sample(n=8000, random_state=2)
#df=full_df

#Choose a random data point, reshape, and plot the image
imageTest=df.iloc[1,1:]
plt.imshow(np.array(imageTest).reshape(28,28))

#identify the features and the target
x=df.iloc[:,1:]
y=df.iloc[:,0]

#split the data into training and testing samples
xtrain, xtest, ytrain, ytest=train_test_split(x,y,test_size=0.2, shuffle=False, random_state=2)

#convert all values to floats before renormalization
xtrain=xtrain.astype('float32')
ytrain=ytrain.astype('float32')

#normalize the features to 0,1
xtrain/=255

#get a numpy array from the pd dataframe of featuers
xtrain_formatted=xtrain.values

#Since we have 10 different classes, we will have 10 outputs 
#So we oneHotEncode the target
ytrain_dummies=pd.get_dummies(ytrain,prefix='dummy')
ytrain_formatted=ytrain_dummies.values


#Since this is a custom library, I am testing the time that it takes to train the model
#Start the timer
seconds1 = time.time()

#Create an instance of the FNN_model building class
model=FFNN_pymodule.FFNN_Builder()
#Set the topology input for the FFNN_model object
#our first layer is the input layer, which is 784 neurons wide because we have 784 pixels
model.setTopology([784, 16, 10])
#We use sigmoid activation becuase it gives an output from 0,1 and it generally performs well with classification (e.g. logistic regression)
model.setActivationFunction('sigmoid')
#Set the number of epochs (i.e the number of times the model runs through all of the data)
#Our library uses stochastic gradient descent so we cannot choose batches
model.setEpochs(1)
#Finaly we fit the model on our training data
model.fitModel(xtrain_formatted, ytrain_formatted)

#Stop the timer and print the total time that it took to train the model
seconds2 = time.time()
print("total time", (seconds2-seconds1)/60)

#Test the model on a random sample
#Choose a random sample from the dataset and preprocess the data
img=np.random.randint(0,2000)
imageTest=full_df.iloc[img,1:]
catTest=full_df.iloc[img,0]
testImage=np.array(imageTest)
testImage=testImage.astype('float32')
testImage/=255

#print the true category
print('Test image:', catTest)

#print the models predictions
print('Prediction:')
print(np.array(model.predict(testImage)).argmax())



#Apply the model to the testing data 
#keep track of our predicted classes
y_true=[]
y_predicted=[]

#loop through all of the testing data, process the data, call the model to make a prediction, save the predictions and the actual values in arrays
for imgindex in range(len(np.array(xtest)[:,0])):
    image=xtest.iloc[imgindex,:]
    trueClass=ytest.iloc[imgindex]
    image=np.array(image)

    image=image.astype('float32')
    image/=255
    predictedClass=np.array(model.predict(image)).argmax()
    y_predicted.append(predictedClass)

    trueClass=ytest.iloc[imgindex]
    y_true.append(trueClass)

#Call the scikitlearn confusion matrix method to generate a confusion matrix based on the actual and predicted values
conMat=confusion_matrix(y_true, y_predicted)
#Generate a heatmap from the confusion matrix with seaborn
ax=sns.heatmap(conMat)
print(conMat)

#Get the accuracy for the model classification
accuracy=accuracy_score(y_true, y_predicted)
print('Model accuracy: ', accuracy)


#We now serialize the model using pickle
#Currently the serialization procedure that I built in C++ just saves 
# the input parameters and the input data and retrains the model when the model is loaded 
# (pointless, I know, but I am working on a better serialization procedure)
pickle.dump(model,open('model_file.p',"wb"), pickle.HIGHEST_PROTOCOL)

















#Here I am just testing my serialized model and timing the model loading+predicting
#I am keeping this here because I am currently int the process of updating the serialization method
seconds1 = time.time()
model2=pickle.load(open('model_file.p','rb'))
print('now from pickle:')
img=np.random.randint(0,20000)
imageTest=full_df.iloc[img,1:]
catTest=full_df.iloc[img,0]
testImage=np.array(imageTest)
testImage=testImage.astype('float32')
testImage/=255
print('test image:', catTest)
print(np.array(model2.predict(testImage)).argmax())
seconds2 = time.time()
print("Pickle time:", (seconds2-seconds1)/60)
