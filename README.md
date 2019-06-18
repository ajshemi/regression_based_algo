# regression_based_algo
Python aglo trading script using regression (linear and logistic) for prediction

oaOLSTrader.py  #python script#

This uses linear regression for prediction.The linear regression model can also be from recent historically data. However, in this script, the linear regression model is updated cumulatively as more streaming data is collected. 


logregTrader.py  #python script#

This uses logistic regression for prediction. The logistic regression model is part of scikit learn machine learning in python. The features are created from the lag of returns. The lags of returns are then bucketize prior to fitting the model. The logistic regression model is updated cumulatively as more streaming data is collected.
