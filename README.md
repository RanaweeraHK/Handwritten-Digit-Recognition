# Handwritten-Digit-Recognition

Read the blog article : https://dev-hashr.pantheonsite.io/handwritten-digit-recognition/

network model using the Sequential API in TensorFlow. The model consists of three layers: a flattening layer, two dense layers with ReLU activation, and an output layer with softmax activation.
mnist = tf.keras.datasets.mnist  # Load MNIST dataset into the code
Here, we're using TensorFlow (tf) to load the MNIST dataset.(https://medium.com/@ranaweerahk/unveiling-the-mnist-dataset-a-journey-into-handwritten-digit-recognition-bcc9c52b68ac) The MNIST dataset is a collection of 28×28 pixel grayscale images of handwritten digits (0 through 9). This dataset is commonly used for training and testing machine learning models.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
We're unpacking the loaded MNIST dataset into training and testing sets. x_train contains the images used for training, and y_train contains the corresponding labels (the actual digits). Similarly, x_test contains images for testing, and y_test contains their labels.
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
Normalization is a process of scaling the pixel values of the images to a range between 0 and 1. This is important for training a neural network as it helps in faster convergence during the training process. In simpler terms, we're making sure all the pixel values are in a consistent and manageable range.
These lines of code set up the MNIST dataset, separate it into training and testing sets, and normalize the pixel values of the images. This pre-processing prepares the data for training a neural network that can learn to recognize handwritten digits.
Building the Neural Network Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model = tf.keras.models.Sequential()
This line initializes a sequential model, which is a linear stack of layers. It's a simple way to build a model layer by layer, where the output of one layer becomes the input to the next.
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
Here, we add the first layer to the model. The Flatten layer is used to flatten the input, which is a 28×28 array representing the image. It transforms the input into a 1D array of size 28 * 28 = 784. This layer serves as the input layer.
model.add(tf.keras.layers.Dense(128, activation='relu'))
We add a dense (fully connected) layer with 128 neurons (units). Each neuron in this layer is connected to every neuron in the previous layer. The activation function used here is Rectified Linear Unit (ReLU), which introduces non-linearity to the model.
model.add(tf.keras.layers.Dense(128, activation='relu'))
Another dense layer with 128 neurons and ReLU activation is added. This introduces more complexity and non-linearity to the model, allowing it to learn intricate patterns in the data.
model.add(tf.keras.layers.Dense(10, activation='softmax'))
The final dense layer has 10 neurons, corresponding to the 10 possible classes (digits 0 through 9). The activation function is softmax, which is often used in the output layer of a multi-class classification model. It converts the raw output values into probabilities for each class.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
After constructing the model, we compile it using the Adam optimizer, sparse categorical crossentropy loss function, and accuracy as the metric. Compilation is a necessary step before training the model, where the optimizer, loss function, and evaluation metric are specified.
This code constructs a simple neural network model with one input layer (flatten), two hidden layers with ReLU activation, and an output layer with softmax activation. The model is compiled with the Adam optimizer for training on the MNIST dataset.
Training the Model
model.fit(x_train, y_train, epochs=3)
model.save('handwrittenRecognition.model')model.save('handwrittenRecognition.model')
model.fit(x_train, y_train, epochs=3)
In this line, you are using the fit method to train your neural network model. The x_train and y_train are your training data and corresponding labels, and epochs=3 indicates that you want to iterate over the entire training dataset three times. During each epoch, the model learns from the training data and adjusts its weights to minimize the loss.
model.save('handwrittenRecognition.model')
After training, you save the trained model using the save method. The argument 'handwrittenRecognition.model' is the file name or path where the model will be saved. This allows you to later load the trained model for making predictions on new data without having to retrain it.
Evaluating the Model
model = tf.keras.models.load_model('handwrittenRecognition.model')
loss, accuracy = model.evaluate(x_test, y_test)

print("Loss : ", loss)
print("Accuracy : ", accuracy)
model = tf.keras.models.load_model('handwrittenRecognition.model')
This line loads the saved model from the specified file. The model should have been saved using the model.save() method in a previous part of the code.
loss, accuracy = model.evaluate(x_test, y_test)
Here, the evaluate method is used to assess the model's performance on the test dataset (x_test and y_test). It returns the loss and accuracy of the model on the test data.
print("Loss : ", loss)
print("Accuracy : ", accuracy)
Finally, the obtained loss and accuracy values are printed to the console. The loss is a measure of how well the model is performing, and the accuracy represents the proportion of correctly classified instances in the test dataset. Higher accuracy and lower loss values generally indicate better model performance.
Output:
You can see that our model has 97% accuracy!!!
Handwritten Digit Prediction
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
In this part of the code, you are iterating through a sequence of images with filenames following the pattern "digit{image_number}.png" in the "HandWritten-Digits" directory. For each image, you perform the following steps:
img = cv2.imread(f"HandWritten-Digits/digit{image_number}.png")[:,:,0]
You can download the HandWritten-Digits folder here: https://github.com/RanaweeraHK/Handwritten-Digit-Recognition
This line reads the image using OpenCV (cv2). It loads the image in color, but you extract only the first channel (assuming it's a grayscale image) using [:,:,0].
img = np.invert(np.array([img]))
This line inverts the pixel values of the image. It converts the image to a NumPy array and applies the np.invert function, which flips the bits of the array, effectively inverting the pixel values. This step is often necessary because models are trained on datasets where the background is dark, and digits are light.
prediction = model.predict(img)
Here, you use the loaded model (model) to make predictions on the preprocessed image.
print("The number is probably a {}".format(np.argmax(prediction)))
You print the predicted digit, which is the index of the maximum value in the prediction array. This assumes that your model is designed to predict digits.
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()
Finally, you display the inverted image using Matplotlib, with a binary color map (cmap=plt.cm.binary) to visualize it as a grayscale image.
The process is repeated for each image in the sequence, and any errors encountered during the process are caught and printed to the console. The iteration continues until there are no more image files matching the specified pattern.
When you run this code, you can see that the model identifies most of the handwritten digits correctly. !!!
In conclusion, our journey through the intricacies of handwritten digit recognition using TensorFlow has provided a comprehensive understanding of the essential steps involved in building and training a neural network model. The MNIST dataset served as our guide, showcasing the power of machine learning in recognizing handwritten digits. From loading and preprocessing the data to constructing a neural network model, our exploration touched upon key aspects of the process. We delved into the significance of model compilation, training, and evaluation, with an emphasis on saving and loading models for practical applications. Through a real-world example, where our model made predictions on a series of handwritten digit images, we witnessed the tangible impact of our efforts. This exploration not only demystifies the fundamentals of image classification but also lays the groundwork for future endeavors in the ever-evolving landscape of machine learning.
Happy coding !!!
