from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten

# Initialize the Sequential model
CNN = Sequential()

# Convolutional and Pooling layers
CNN.add(Conv2D(input_shape=(128, 128, 1), filters=32, kernel_size=(3, 3), padding="same", activation='relu'))
CNN.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
CNN.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
CNN.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
CNN.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
CNN.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))

# Flattening the vectors
CNN.add(Flatten())

# Dense layers
CNN.add(Dense(units=128, activation='relu'))
CNN.add(Dense(units=64, activation='relu'))
CNN.add(Dense(units=4, activation='softmax'))

