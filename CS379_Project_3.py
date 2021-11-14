##Adam Mick CS379 Project 3

import pandas as pd #reading data
import numpy as np
from sklearn.preprocessing import LabelEncoder #data transformation
from sklearn.ensemble import RandomForestClassifier #random forest algorithm to be implemented
from sklearn.model_selection import train_test_split# Using Skicit-learn to split data into training and testing sets
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix #imports for testing the algorithm


def main():

    ###############
    #Load the Data#
    ###############

    data = pd.read_csv('dataset.csv') #read the data

    data.drop('location', axis = 1, inplace = True) #removing useless data
    data.drop('other_parties', axis = 1, inplace = True)
    data.drop('housing', axis = 1, inplace = True)
    data.drop('own_telephone', axis = 1, inplace = True)
    data.drop('foreign_worker', axis = 1, inplace = True)
    data.drop('personal_status', axis = 1, inplace = True)
    data.drop('id', axis = 1, inplace = True)
    data.drop('purpose', axis = 1, inplace = True)


    ####################
    #Transform the data#
    ####################
    lbl = LabelEncoder()
    data = data.apply(lbl.fit_transform) #using label encoder to transform all of the data to numerical data so it can run across the algoritm
   
    ####################################
    #Calculating fradulnet transactions#
    ####################################

    Fraud = 0
    i = 0
    while i < len(data['class']):
        if data['class'][i]==0 and data['job'][i] == 3: #transactions that are bad credit standing & unemployed are flagged as fradulent
            Fraud = Fraud + 1
        i = i + 1

    i = 0
    while i < len(data['savings']):
        if data['savings'][i]==3 and data['checking'][i] == 1: #transactions that have no checking or savings account flag as fraud
            Fraud = Fraud + 1
        i = i + 1

    Valid = data[data['class'] == 1] #with just the parameters above .35% of the transactions are fradulent
    outlier_fraction = Fraud/float(len(Valid))
    print (outlier_fraction) #calculating fradulent transaction precentage
    print('Fraud Cases: ', Fraud)
    print('Valid Transactions: {}'.format(len(data[data['class'] == 1])))


    ########################
    #Training the algorithm#
    ########################

    X = data.drop(['class'], axis = 1)
    Y = data['class']
    X_data = X.values
    Y_data = Y.values


    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 42)

    forest = RandomForestClassifier()
    # random forest model creation
    forest = RandomForestClassifier()
    forest.fit(X_train,Y_train)
    # predictions
    y_pred = forest.predict(X_test)

    
    ################################
    #Testing the algorithm accuarcy#
    ################################


    n_outliers = Fraud
    n_errors = (y_pred != Y_test).sum()
    print("The model used is Random Forest classifier")
    acc= accuracy_score(Y_test,y_pred)
    print("The accuracy is {}".format(acc))
    prec= precision_score(Y_test,y_pred)
    print("The precision is {}".format(prec))
    rec= recall_score(Y_test,y_pred)
    print("The recall is {}".format(rec))
    f1= f1_score(Y_test,y_pred)
    print("The F1-Score is {}".format(f1))
    MCC=matthews_corrcoef(Y_test,y_pred)
    print("The Matthews correlation coefficient is ", MCC)

    return

main()
