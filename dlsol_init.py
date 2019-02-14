import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


train = pd.read_csv('train_LZdllcl.csv')
test = pd.read_csv('test_2umaH9m.csv')



X_train = train.iloc[:,1:13]
Y_train = train.iloc[:,13]

X_test = test.iloc[:,1:13]

#X = train.iloc[:,1:13]
#Y = train.iloc[:,13]

#X_test = test.iloc[:,1:13]


#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 1)

X_train["department"].value_counts()
X_train["region"].value_counts()
X_train["education"].value_counts()
X_train["gender"].value_counts()
X_train["recruitment_channel"].value_counts()

X_train = X_train.fillna({"department": "Sales & Marketing"})
X_train = X_train.fillna({"gender": 'm'})
X_train = X_train.fillna({"education": 'Bachelor\'s'})
X_train = X_train.fillna({"region": "region_2"})
X_train = X_train.fillna({"recruitment_channel": "other"})
X_train["no_of_trainings"] = X_train["no_of_trainings"].fillna(X_train["no_of_trainings"].mean())
X_train["previous_year_rating"] = X_train["previous_year_rating"].fillna(X_train["previous_year_rating"].mean())
X_train["age"] = X_train["age"].fillna(X_train["age"].mean())
X_train["length_of_service"] = X_train["length_of_service"].fillna(X_train["length_of_service"].mean())
X_train["KPIs_met >80%"] = X_train["KPIs_met >80%"].fillna(X_train["KPIs_met >80%"].mean())
X_train["awards_won?"] = X_train["awards_won?"].fillna(X_train["awards_won?"].mean())
X_train["avg_training_score"] = X_train["avg_training_score"].fillna(X_train["avg_training_score"].mean())

#Test

X_test["department"].value_counts()
X_test["region"].value_counts()
X_test["education"].value_counts()
X_test["gender"].value_counts()
X_test["recruitment_channel"].value_counts()


X_test = X_test.fillna({"department": "Sales & Marketing"})
X_test = X_test.fillna({"gender": 'm'})
X_test = X_test.fillna({"education": 'Bachelor\'s'})
X_test = X_test.fillna({"region": "region_2"})
X_test = X_test.fillna({"recruitment_channel": "other"})
X_test["no_of_trainings"] = X_test["no_of_trainings"].fillna(X_test["no_of_trainings"].mean())
X_test["previous_year_rating"] = X_test["previous_year_rating"].fillna(X_test["previous_year_rating"].mean())
X_test["age"] = X_test["age"].fillna(X_test["age"].mean())
X_test["length_of_service"] = X_test["length_of_service"].fillna(X_test["length_of_service"].mean())
X_test["KPIs_met >80%"] = X_test["KPIs_met >80%"].fillna(X_test["KPIs_met >80%"].mean())
X_test["awards_won?"] = X_test["awards_won?"].fillna(X_test["awards_won?"].mean())
X_test["avg_training_score"] = X_test["avg_training_score"].fillna(X_test["avg_training_score"].mean())






#Train
X_train.loc[X_train['recruitment_channel'] == 'other','recruitment_channel'] = 0
X_train.loc[X_train['recruitment_channel'] == 'sourcing','recruitment_channel'] = 1
X_train.loc[X_train['recruitment_channel'] == 'referred','recruitment_channel'] = 2

X_train.loc[X_train['gender'] == 'm','gender'] = 0
X_train.loc[X_train['gender'] == 'f','gender'] = 1

X_train.loc[X_train['education'] == 'Bachelor\'s','education'] = 0
X_train.loc[X_train['education'] == 'Master\'s & above','education'] = 1
X_train.loc[X_train['education'] == 'Below Secondary','education'] = 2

i = 0
for elements in set(X_train['region']):
    X_train.loc[X_train['region'] == elements,'region'] = i
    i+=1

i = 0
for elements in set(X_train['department']):
    X_train.loc[X_train['department'] == elements,'department'] = i
    i+=1

#Test

X_test.loc[X_test['recruitment_channel'] == 'other','recruitment_channel'] = 0
X_test.loc[X_test['recruitment_channel'] == 'sourcing','recruitment_channel'] = 1
X_test.loc[X_test['recruitment_channel'] == 'referred','recruitment_channel'] = 2

X_test.loc[X_test['gender'] == 'm','gender'] = 0
X_test.loc[X_test['gender'] == 'f','gender'] = 1

X_test.loc[X_test['education'] == 'Bachelor\'s','education'] = 0
X_test.loc[X_test['education'] == 'Master\'s & above','education'] = 1
X_test.loc[X_test['education'] == 'Below Secondary','education'] = 2

i = 0
for elements in set(X_test['region']):
    X_test.loc[X_test['region'] == elements,'region'] = i
    i+=1

i = 0
for elements in set(X_test['department']):
    X_test.loc[X_test['department'] == elements,'department'] = i
    i+=1



#0,1,2,3,4
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [0])
X_train = onehotencoder.fit_transform(X_train).toarray()

onehotencoder = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder.fit_transform(X_train).toarray()

onehotencoder = OneHotEncoder(categorical_features = [2])
X_train = onehotencoder.fit_transform(X_train).toarray()

onehotencoder = OneHotEncoder(categorical_features = [3])
X_train = onehotencoder.fit_transform(X_train).toarray()

onehotencoder = OneHotEncoder(categorical_features = [4])
X_train = onehotencoder.fit_transform(X_train).toarray()


#Test
onehotencoder = OneHotEncoder(categorical_features = [0])
X_test = onehotencoder.fit_transform(X_test).toarray()

onehotencoder = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder.fit_transform(X_test).toarray()

onehotencoder = OneHotEncoder(categorical_features = [2])
X_test = onehotencoder.fit_transform(X_test).toarray()

onehotencoder = OneHotEncoder(categorical_features = [3])
X_test = onehotencoder.fit_transform(X_test).toarray()

onehotencoder = OneHotEncoder(categorical_features = [4])
X_test = onehotencoder.fit_transform(X_test).toarray()



from sklearn.preprocessing import StandardScaler
#Feature scaling
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.fit_transform(X_test)



#importing keras
import keras
from keras.models import Sequential    #Used to initialise neural net
from keras.layers import Dense         #Used for hidden layers

'''Two methods for initialising 1. defining as sequence of layers
2. defining as graph
here we are using sequence of layers'''
#Initialising ANN
classifier = Sequential()

'''Generally rectifier activation function is preferred for hidden layer and in output layer sigmoid activation function
is preferred as it provides probabilities'''

'''TIP: Choose number of nodes of hidden layer equal to average f number of nodes in input and out put layers.
Or we can use parameter tuning'''

'''Arguements:
    output_dim : No of nodes in hidden layer , init : weight initial,  activation: activation function used
    in first step input dim is a compulsory arguement because we have just intialised the neural net, 
    input_dim:no of independent variables'''
    
'''If dependent variable has more than two categories then activation fn is softmax for output layer'''

#Adding input layer and the first hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu',input_dim = 24))

#Adding second hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

#Adding output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'softmax'))

'''Compiling means applying stochastic gradient descent on whole ANN
Arguements :
    optimizer: adam is efficient form of stochastic gradient descent
    loss : binary for two output, for more than two categorical
    metrics : criterion for improving models performence'''
#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

'''epochs means number of rounds
   also we have to add the batch size '''

#Fitting ANN to the dataset
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 15)


ids = test['employee_id']
    
Y_pred = classifier.predict(X_test)
Y_pred = ((Y_pred >0.25)*1)
Y_pred = Y_pred.ravel()


#sklearn.metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
#cm = confusion_matrix(Y_test, Y_pred)

submission = pd.DataFrame({'employee_id': ids,
                           'is_promoted':Y_pred
                           })
#predictions = pd.DataFrame(Y_pred, columns=['is_promoted']).to_csv('prediction.csv')
submission.to_csv("pred.csv",index=False)