import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import random

# plt playground
def show_rand_img():
    n = random.randint(0, 59999)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[n], cmap=plt.cm.binary)
    plt.xlabel(f"{n} {class_names[train_labels[n]]}")

def show_25_img():
    for i in range(25):
        plt.subplot(5, 5, i+1)
        show_rand_img()

def plot_img(i, predict_arr, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[i], cmap=plt.cm.binary)

    if np.argmax(predict_arr) == true_label[i]:
        color = "blue"
    else:
        color = "red"

    plt.xlabel("{} {:.2f} ({})".format(class_names[np.argmax(predict_arr)],
                                        100*np.max(predict_arr),
                                        class_names[true_label[i]]),
                                        color=color)

def plot_value_array(i, predict_arr, true_label):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    plot = plt.bar(range(10), predict_arr, color="#777777")
    plt.ylim([0, 1])
    
    plot[np.argmax(predict_arr)].set_color("red")
    plot[true_label[i]].set_color("blue")

print("========================")
print("tensorflow version", tf.__version__)
print("========================")

fashion_mnist = keras.datasets.fashion_mnist

# train_images.shape => (60000, 28, 28)
# test_images.shape => (10000, 28, 28)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# setting up value 0-255 => 0-1
train_images = train_images / 255
test_images = test_images / 255

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("train_images.shape :", train_images.shape)
print("test_iamges.shpae  :", test_images.shape)

plt.figure("Example images", figsize=(10, 10))
show_25_img()
plt.tight_layout()
plt.show()

# set up the layers
model = keras.Sequential(layers=[
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
], name="NeuralNetwork")

model.summary()

# complie model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# training model
model.fit(train_images, train_labels, epochs=10)

# evaluate model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f"test loss: {test_loss}")
print(f"test accuracy: {test_acc}")

# prediction model
probability_model = tf.keras.Sequential(name="Prediction", layers=[model, 
                                         tf.keras.layers.Softmax()])
probability_model.summary()
predictions = probability_model.predict(test_images)

print(np.argmax(predictions[0]))
print(test_labels[0])

plt.figure("Tested", figsize=(12, 10))
for i in range(15):
    plt.subplot(5, 6, 2*i+1)
    plot_img(i, predictions[i], test_labels, test_images)
    plt.subplot(5, 6, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# using probability model
img = test_images[1]
print("img shape :", img.shape)
img = (np.expand_dims(img,0))
print("img shape :", img.shape)

single_prediction = probability_model(img)

plt.figure("using prediction model")
plt.subplot(1, 2, 1)
plot_img(1, single_prediction[0], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(1, single_prediction[0], test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.tight_layout()
plt.show()