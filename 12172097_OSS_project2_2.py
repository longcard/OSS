import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def sort_dataset(dataset_df):
    #TODO: Implement this function
    result = dataset_df.sort_values(by='year')
    return result

def split_dataset(dataset_df):	
	#TODO: Implement this function
	X = dataset_df.drop(columns='salary', axis=1)
	y = dataset_df['salary'] * 0.001
	X_train = X[:1718]
	y_train = y[:1718]
	X_test = X[1718:]
	y_test = y[1718:]
	return X_train, X_test, y_train, y_test


def extract_numerical_cols(dataset_df):
	#TODO: Implement this function
	result = dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]
	return result

def train_predict_decision_tree(X_train, Y_train, X_test):
	#TODO: Implement this function
	dt_pipe = make_pipeline(
		StandardScaler(),
  		DecisionTreeRegressor()
	)
	dt_pipe.fit(X_train, Y_train)
	prediction = dt_pipe.predict(X_test)
	return prediction

def train_predict_random_forest(X_train, Y_train, X_test):
	#TODO: Implement this function
	rf_pipe = make_pipeline(
		StandardScaler(),
  		RandomForestRegressor()
	)
	rf_pipe.fit(X_train, Y_train)
	prediction = rf_pipe.predict(X_test)
	return prediction

def train_predict_svm(X_train, Y_train, X_test):
	#TODO: Implement this function
	svm_pipe = make_pipeline(
		StandardScaler(),
  		SVR()
	)
	svm_pipe.fit(X_train, Y_train)
	prediction = svm_pipe.predict(X_test)
	return prediction

def calculate_RMSE(labels, predictions):
	#TODO: Implement this function
	result = np.sqrt(np.mean((predictions - labels)**2))
	return result

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))