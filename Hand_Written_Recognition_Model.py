import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


mnist = tf.keras.datasets.mnist  # Load MNIST dataset into the code

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('handwrittenRecognition.model')

# After this, you can comment the above code if not all the time the model goes through the dataset.

model = tf.keras.models.load_model('handwrittenRecognition.model')

loss, accuracy = model.evaluate(x_test, y_test)

print("Loss : ", loss)
print("Accuracy : ", accuracy)

image_number = 1
while os.path.isfile(f"HandWritten-Digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"HandWritten-Digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error reading image! Proceeding to the next one...")
    finally:
        image_number += 1
