# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
data = pd.read_csv('RELIANCE_EQN.csv')
dataset = data.iloc[:, 4:5 ].values

training_set = dataset[ :376 , :] 
test_set = dataset[ 376: , :]

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(15, training_set.shape[0]):
    X_train.append(training_set_scaled[i-15:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# print(X_train.shape)
# print(y_train.shape)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# # Importing the Keras libraries and packages
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout

# # Initialising the RNN
# regressor = Sequential()

# # Adding the first LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# regressor.add(Dropout(0.2))

# # Adding a second LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 50, return_sequences = True))
# regressor.add(Dropout(0.2))

# # Adding a third LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 50, return_sequences = True))
# regressor.add(Dropout(0.2))

# # Adding a fourth LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 50))
# regressor.add(Dropout(0.2))

# # Adding the output layer
# regressor.add(Dense(units = 1))

# # Compiling the RNN
# regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# # Fitting the RNN to the Training set
# regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# regressor.save('RIL.model')

from tensorflow.keras.models import load_model
regressor = load_model('RIL.model')

# Part 3 - Making the predictions and visualising the results
inputs = dataset[dataset.shape[0] - test_set.shape[0] - 15:, :]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(15, 15 + test_set.shape[0]):
    X_test.append(inputs[i-15:i, :])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
predicted_data = regressor.predict(X_test)
predicted_data = sc.inverse_transform(predicted_data)


# Visualising the results
plt.plot(test_set[: , 0], color = 'red', label = 'Real Reliance Stock Price')
plt.plot(predicted_data[:, 0], color = 'blue', label = 'Predicted Reliance Stock Price')
plt.title('Reliance Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Reliance Stock Price')
plt.legend()
plt.show()
