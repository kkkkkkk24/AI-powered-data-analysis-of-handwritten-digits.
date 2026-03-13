from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical

# Load dataset
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

# Check shape
print(train_images.shape)
print(test_images.shape)

# Show sample image
plt.imshow(train_images[1], cmap='gray')
plt.show()

print(train_labels[1])

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.

train_images = train_images.reshape((60000,28,28,1))
test_images = test_images.reshape((10000,28,28,1))

# Reshape for CNN
train_images = train_images.reshape((60000,28,28,1))
test_images = test_images.reshape((10000,28,28,1))

# One hot encode labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build CNN
cnnmodel = Sequential()

cnnmodel.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
cnnmodel.add(MaxPooling2D(2,2))

cnnmodel.add(Conv2D(64,(3,3),activation='relu'))
cnnmodel.add(MaxPooling2D(2,2))

cnnmodel.add(Conv2D(64,(3,3),activation='relu'))

cnnmodel.add(Flatten())

cnnmodel.add(Dense(64,activation='relu'))

cnnmodel.add(Dense(10,activation='softmax'))

from keras.layers import Dropout
cnnmodel.add(Dense(64,activation='relu'))
cnnmodel.add(Dropout(0.3))
cnnmodel.add(Dense(10,activation='softmax'))

# Compile
cnnmodel.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Train
cnnmodel.fit(train_images,train_labels,epochs=10)

# Predict
predictions=cnnmodel.predict(test_images)

# Compare predictions
for i in range(3):
  print("Prediction:", predictions[i])
  print("Actual:", test_labels[i])

  test_loss, test_acc = cnnmodel.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)
cnnmodel.save("digit_model.h5")

test_loss, test_acc = cnnmodel.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)