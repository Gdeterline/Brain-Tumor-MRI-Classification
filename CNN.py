from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

# Initialize the Sequential model
Modified_VGG16 = Sequential()

# Convolutional and Pooling layers
Modified_VGG16.add(Conv2D(input_shape=(224, 224, 1), filters=64, kernel_size=(3, 3), padding="same", activation='relu'))
Modified_VGG16.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
Modified_VGG16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
Modified_VGG16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
Modified_VGG16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
Modified_VGG16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
Modified_VGG16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
Modified_VGG16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
Modified_VGG16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
Modified_VGG16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
Modified_VGG16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
Modified_VGG16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
Modified_VGG16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
Modified_VGG16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
Modified_VGG16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
Modified_VGG16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
Modified_VGG16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
Modified_VGG16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Flattening the vectors
Modified_VGG16.add(Flatten())

# Dense layers
Modified_VGG16.add(Dense(units=4096, activation='relu'))
Modified_VGG16.add(Dense(units=4096, activation='relu'))
Modified_VGG16.add(Dense(units=4, activation='softmax'))


