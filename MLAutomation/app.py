#previous https://www.programiz.com/python-programming/datetime/current-datetime
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
print(timestr)


# Train the model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np

import pandas as pd

# load the training dataset
bike_data = pd.read_csv('https://raw.githubusercontent.com/MicrosoftDocs/ml-basics/master/data/daily-bike-share.csv')
bike_data.head()

# Separate features and labels
X, y = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values, bike_data['rentals'].values
#print('Features:',X[:10], '\nLabels:', y[:10], sep='\n')

from sklearn.model_selection import train_test_split

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#Experiment size log
#file name
filename = "logs/"+ timestr + ".txt"

f = open(filename, "a")
f.write('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))
f.close()

#open and read the file after the appending:
f = open(filename, "r")
print(f.read())

# Train the model
from sklearn.linear_model import LinearRegression

# Fit a linear regression model on the training set
model = LinearRegression().fit(X_train, y_train)
print (model)

#run predictions
predictions = model.predict(X_test)
np.set_printoptions(suppress=True)
print('Predicted labels: ', np.round(predictions)[:10])
print('Actual labels   : ' ,y_test[:10])

#Plot scatter plot to see fit
import matplotlib.pyplot as plt

%matplotlib inline

plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.savefig("logs/" + timestr + "jpg")
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)

rmse = np.sqrt(mse)
print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)
print("R2:", r2)

f = open(filename, "a")
f.write('\nMSE: %2f \nRMSE: %2f \nR2: %2f'% (mse, rmse, r2))
f.close()

import joblib
# Save the model as a pickle file
filename = 'models/' + timestr + ".pkl"
joblib.dump(model, filename)
