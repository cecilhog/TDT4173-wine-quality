import os
import argparse
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, confusion_matrix

def load_dataset(wine_type):
    #Returns either the red or white dataset as a Pandas dataframe
    
    #Dataset URLs (public sources)
    red_dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    white_dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    
    #Return dataset associated with specified wine type, or comment on error
    if wine_type == 'red':
        return pd.read_csv(red_dataset_url, sep=';')
    elif wine_type == 'white':
        return pd.read_csv(white_dataset_url, sep=';')
    else:
        raise Exception("Invalid wine type. Must  be either 'red' or 'white'")

def prepare_datasets(wine_type, scaler_type):
    #Prepares dataset with scaling
    
    #Load dataset
    dataset = load_dataset(wine_type)
    
    #Remove residual sugar extreme outlier from white dataset
    if wine_type == 'white':
        dataset = dataset.drop(dataset['residual sugar'].idxmax(), axis=0)
    
    #Split dataset in X (features) and y (quality score)
    y = dataset.quality
    X = dataset.drop('quality', axis=1)
    
    #Split datasets into stratified training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8, stratify=y)
    
    #Scale features
    if scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaler_type == 'None':
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
    else:
        raise Exception("Invalid scaler")
    
    #Convert target to numpy arrays
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    return X_train, X_test, y_train, y_test

def binarize_y(y, cutoff):
    #Iterates through the array and sets quality score to either 0 or 1 depending on the quality cutoff

    for i in range(len(y)):
        if y[i] >= cutoff:
            y[i] = 1
        else:
            y[i] = 0
    return y

def remove_features(X, wine_type, index):
    #Removes features from the dataset
    
    #Indices of columns/attributes to remove from the dataset
    if wine_type == 'white':
        if index == 1:
            arg = [2,3,5,8,9]  #Removes features with correlation with quality between -0.10 and 0.10 ('citric acid','residual sugar', 'free sulfur dioxide', 'pH', 'sulphates')
        elif index == 2:
            arg = [2,5,9] #Removes 3 features with least correlation with quality ('citric acid', 'free sulfur dioxide' 'sulphates')
        elif index == 3:
            arg = [5,7] #Removes features with highest pairwise correlation (above 0.5 or below -0.5), ('free sulfur dioxide', 'density')
        else: 
            print('Invalid index')
    elif wine_type == 'red':
        if index == 1:
            arg = [3, 5, 8] #Removes features with correlation with quality between -0.10 and 0.10 ('residual sugar','free sulfur dioxide', 'pH')
        elif index == 2:
            arg = [2, 5, 7, 8] #Removes features with highest pairwise correlation (above 0.5 or below -0.5), ('citric acid', 'free sulfur dioxide', 'density', 'pH')
        elif index  == 3:
            arg = [0] #Removes 'fixed acidity'
        else: 
            print('Invalid index')
    else:
        print('Invalid wine type')
        
    X = np.delete(X, arg, axis=1)

    return X

def define_scoring(target_val):
    #Set scoring parameter used in GridSearchCV, depending on whether target is multiclass or binary
    
    if target_val == 'binary':
        scoring = 'f1'
    elif target_val == 'multiclass':
        scoring = 'neg_mean_squared_error' #Negated because GridSearchCV optimizes so that high scores are better than low scores
    else:
        raise Exception('Invalid target values')
        
    return scoring

def report_performance(y_test, y_pred, target_val):
    #Calculates and prints test performance
    
    print('Accuracy:     \t', accuracy_score(y_test, y_pred))
    print('Average error:\t', mean_absolute_error(y_test, y_pred))
    
    if target_val == 'binary':
        print('F1-score:     \t', f1_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
    elif target_val == 'multiclass':
        print('MSE:          \t', mean_squared_error(y_test, y_pred))
        
def correlation_matrix(dataset, wine_type):
    #Inspired by code from https://www.kaggle.com/robjan/wine-quality-prediction-with-visualisation#Plotting-graphs---plotly
    
    #Create correlation matrix and save as .png
    corr_matrix = dataset.corr()
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(corr_matrix,annot=True,linewidths=0.5,ax=ax)
    plt.subplots_adjust(bottom=0.28, left=0.20)
    plt.savefig('correlation_matrix_' + wine_type + '.png')
    print("Figure saved as 'correlation_matrix_" + wine_type + ".png'")
    
    #Print features correlated to quality in descending order
    print(corr_matrix['quality'].sort_values(ascending=False))
        
if __name__ == '__main__':
    
    #Structure to test different parts of the code
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question
    
    if question == 'test':  
        #Dummy
        print('test')
        
    elif question == 'Correlation': 
        #Creates heatmaps visualizing feature correlation, both for the red and white dataset
        #Works on pandas dataframes
        
        red_dataset = load_dataset('red')
        correlation_matrix(red_dataset, 'red')
        
        white_dataset = load_dataset('white')
        #Remove outlier
        white_dataset = white_dataset.drop(white_dataset['residual sugar'].idxmax(), axis=0)
        correlation_matrix(white_dataset, 'white')
    
    elif question == 'KNN':
        #Define wine dataset to be used, either 'white' or 'red'
        wine_type = 'white'
        print(wine_type)
        
        #Prepare scaled dataset
        X_train, X_test, y_train, y_test = prepare_datasets(wine_type, 'MinMaxScaler')
        
        #Binarize target or not (uncomment to activate)
        target_val = 'multiclass'
        """
        target_val = 'binary'
        cutoff = 7
        print(target_val, cutoff)
        y_train = binarize_y(y_train, cutoff)
        y_test = binarize_y(y_test, cutoff)
        #"""

        #Remove features or not (uncomment to activate)
        """
        index = 1 # valid indices are 1, 2, 3, see function remove_features(X, wine_type, index) for specification
        print('index', index)
        X_train = remove_features(X_train, wine_type, index)
        X_test = remove_features(X_test, wine_type, index)
        #"""
        
        #Create model
        model = KNeighborsClassifier()
        
        #Define parameters to test in GridSearchCV
        k_range = np.array([1,2,3,4,5,10,15,20,30,50])
        w_range = np.array(['distance', 'uniform'])
        p_range = np.array([1,2])
        parameters = dict(n_neighbors=k_range, weights=w_range, p=p_range)
        
        #Define scoring metrics
        scoring = define_scoring(target_val)
        
        #Run cross-validation
        clf = GridSearchCV(model, parameters, cv=5, scoring=scoring, refit=scoring, n_jobs=-1, verbose=0)
        clf.fit(X_train, y_train)
        
        #Remember best model
        best_model = clf.best_estimator_
        
        #Print parameters for best model
        print('Best parameters:\t', clf.best_params_)
        print('Best estimator: \t', best_model)
        
        #Apply model on test data
        y_pred = best_model.predict(X_test)
        
        #Report performance on test data
        report_performance(y_test, y_pred, target_val)
        
    elif question == 'KNN PCA':
        #Set wine dataset to be used, either 'white' or 'red'
        wine_type = 'white'
        print(wine_type)
        
        #Prepare scaled dataset
        X_train, X_test, y_train, y_test = prepare_datasets(wine_type, 'MinMaxScaler')
        
        #Binarize target or not (uncomment to activate)
        target_val = 'multiclass'
        """
        target_val = 'binary'
        cutoff = 7
        print(target_val, cutoff)
        y_train = binarize_y(y_train, cutoff)
        y_test = binarize_y(y_test, cutoff)
        #"""
        
        for i in range(1,12):
            comp = i
            #Create i PCs based on training data
            pca = PCA(n_components=comp).fit(X_train)
            
            #Report variance explained by i PCs
            print("Variance explained", np.sum(pca.explained_variance_ratio_))
            print('pca', comp)
            
            #Transform both train and test data using new basis
            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)

            #Create model
            model = KNeighborsClassifier()

            #Define parameters to test in GridSearchCV
            k_range = np.array([1,2,3,4,5,10,15,20,30,50])
            w_range = np.array(['distance', 'uniform'])
            p_range = np.array([1,2])
            parameters = dict(n_neighbors=k_range, weights=w_range, p=p_range)

            #Define scoring metrics
            scoring = define_scoring(target_val)

            #Run cross-validation
            clf = GridSearchCV(model, parameters, cv=5, scoring=scoring, refit=scoring, n_jobs=-1, verbose=0)
            clf.fit(X_train_pca, y_train)

            #Remember best model
            best_model = clf.best_estimator_

            #Print parameters for best model
            print('Best parameters:\t', clf.best_params_)
            print('Best estimator: \t', best_model)

            #Apply model on test data
            y_pred = best_model.predict(X_test_pca)

            #Report performance on test data
            report_performance(y_test, y_pred, target_val)
 

    elif question == 'Tree':
        #Set wine dataset to be used, either 'white' or 'red'
        wine_type = 'white'
        print(wine_type)
        
        #Prepare scaled dataset
        X_train, X_test, y_train, y_test = prepare_datasets(wine_type, 'MinMaxScaler')
        
        #Binarize target or not (uncomment to activate)
        target_val = 'multiclass'
        """
        target_val = 'binary'
        cutoff = 7
        print(target_val, cutoff)
        y_train = binarize_y(y_train, cutoff)
        y_test = binarize_y(y_test, cutoff)
        #"""

        #Remove features or not (uncomment to activate)
        """
        index = 1
        print('index', index)
        X_train = remove_features(X_train, wine_type, index)
        X_test = remove_features(X_test, wine_type, index)
        #"""
        
        #Create model
        model = DecisionTreeClassifier()
        
        #Define parameters to test in GridSearchCV
        s_range = np.array(['best', 'random'])
        d_range = np.array([5, 10, 15, 20, 30, None])
        parameters = dict(splitter=s_range, max_depth=d_range)
        
        #Define scoring metrics
        scoring = define_scoring(target_val)
        
        #Run cross-validation
        clf = GridSearchCV(model, parameters, cv=5, scoring=scoring, refit=scoring, n_jobs=-1, verbose=0)
        clf.fit(X_train, y_train)
        
        #Remember best model
        best_model = clf.best_estimator_
        
        #Print parameters for best model
        print('Best parameters:\t', clf.best_params_)
        print('Best estimator: \t', best_model)
        
        #Apply model on test data
        y_pred = best_model.predict(X_test)
        
        #Report performance on test data
        report_performance(y_test, y_pred, target_val)

    elif question == 'Tree PCA':
        #Set wine dataset to be used, either 'white' or 'red'
        wine_type = 'white'
        print(wine_type)
        
        #Prepare scaled dataset
        X_train, X_test, y_train, y_test = prepare_datasets(wine_type, 'MinMaxScaler')
        
        #Binarize target or not (uncomment to activate)
        target_val = 'multiclass'
        """
        target_val = 'binary'
        cutoff = 7
        print(target_val, cutoff)
        y_train = binarize_y(y_train, cutoff)
        y_test = binarize_y(y_test, cutoff)
        #"""
        
        for i in range(1,12):
            comp = i
            #Create i PCs based on train data
            pca = PCA(n_components=comp).fit(X_train)
            
            #Report variance explained
            print("Variance explained", np.sum(pca.explained_variance_ratio_))
            print('pca', comp)
            
            #Transform both training and test data to new basis
            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)

            #Create model
            model = DecisionTreeClassifier()

            #Define parameters to test in GridSearchCV
            s_range = np.array(['best', 'random'])
            d_range = np.array([5, 10, 15, 20, 30, None])
            parameters = dict(splitter=s_range, max_depth=d_range)

            #Define scoring metrics
            scoring = define_scoring(target_val)

            #Run cross-validation
            clf = GridSearchCV(model, parameters, cv=5, scoring=scoring, refit=scoring, n_jobs=-1, verbose=0)
            clf.fit(X_train, y_train)

            #Remember best model
            best_model = clf.best_estimator_

            #Print parameters for best model
            print('Best parameters:\t', clf.best_params_)
            print('Best estimator: \t', best_model)

            #Apply model on test data
            y_pred = best_model.predict(X_test)

            #Report performance on test data
            report_performance(y_test, y_pred, target_val)
        
    elif question == 'MLP':
        #Set wine dataset to be used, either 'white' or 'red'
        wine_type = 'white'
        print(wine_type)
        
        #Prepare scaled dataset
        X_train, X_test, y_train, y_test = prepare_datasets(wine_type, 'MinMaxScaler')
        
        #Binarize target or not (uncomment to activate)
        target_val = 'multiclass'
        """
        target_val = 'binary'
        cutoff = 7
        print(target_val, cutoff)
        y_train = binarize_y(y_train, cutoff)
        y_test = binarize_y(y_test, cutoff)
        #"""

        #Remove features or not (uncomment to activate)
        """
        index = 1
        print('index', index)
        X_train = remove_features(X_train, wine_type, index)
        X_test = remove_features(X_test, wine_type, index)
        #"""
        
        #Create model
        model = MLPClassifier(max_iter=1000)
        
        #Define parameters to test in GridSearchCV
        h_range = np.array([(8,), (8,8), (8,8,8)])
        act_range = np.array(['relu', 'logistic'])
        parameters = dict(hidden_layer_sizes=h_range, activation=act_range)
        
        #Define scoring metrics
        scoring = define_scoring(target_val)
        
        #Run cross-validation
        clf = GridSearchCV(model, parameters, cv=5, scoring=scoring, refit=scoring, n_jobs=-1, verbose=0)
        clf.fit(X_train, y_train)
        
        #Remember best model
        best_model = clf.best_estimator_
        
        #Print parameters for best model
        print('Best parameters:\t', clf.best_params_)
        print('Best estimator: \t', best_model)
        
        #Apply model on test data
        y_pred = best_model.predict(X_test)
        
        #Report performance on test data
        report_performance(y_test, y_pred, target_val)
        
    elif question == 'baseline':
        #Set wine dataset to be used, either 'white' or 'red'
        wine_type = 'white'
        print(wine_type)
        
        #Prepare scaled dataset
        X_train, X_test, y_train, y_test = prepare_datasets(wine_type, 'MinMaxScaler')
        
        #Instantiate dummy classifier making predictions based on class distribution in the training set
        dummy = DummyClassifier(strategy='stratified')
        
        #Multi-class case
        #Run 5 iterations to get better estimate of dummy classifier performance
        multiclass_results = np.zeros(5)
        for i in range(5):
            #Fit dummy classifier based on training data
            dummy.fit(X_train, y_train)
            y_pred = dummy.predict(X_test)
            #Calculate mean squared error
            multiclass_results[i] = mean_squared_error(y_test, y_pred)
        #Report mean MSE and standard deviation
        print("Multiclass MSE mean and standard deviation:\t", np.mean(multiclass_results), '\t', np.std(multiclass_results))
    
        #Binary case
        #Run 5 iterations to get better estimate of dummy classifier performance
        cutoff = 7
        y_train = binarize_y(y_train, cutoff)
        y_test = binarize_y(y_test, cutoff)
        
        binary_results = np.zeros(5)
        for i in range(5):
            #Fit dummy classifier based on training data
            dummy.fit(X_train, y_train)
            y_pred = dummy.predict(X_test)
            #Calculate F1-score 
            binary_results[i] = f1_score(y_test, y_pred)
        print("Binary F1-score mean and standard deviation:\t", np.mean(binary_results), '\t', np.std(binary_results))
        
    else:
        print("Unknown question: %s" % question)    