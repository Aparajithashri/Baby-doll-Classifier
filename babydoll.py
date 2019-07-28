import cv2 
import os 
import numpy as np 
from random import shuffle 
import tensorflow as tf
import tflearn
import matplotlib.pyplot as plt
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
im='/content/drive/My Drive/Colab Notebooks/classifier/'
i='/content/drive/My Drive/Colab Notebooks/classifier_test/'
TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 50
LR = 0.3
MODEL_NAME = 'dogs-vs-cats-convnet'
def label(im):
  word=im[51:55]
 
  if word == 'baby':
    return [1,0]
  elif word == 'doll':
    return [0,1]
a=[]

def create_train_data():
    training_data = []
    for f in os.listdir(im):
	      a.append(os.path.join(im,f))
    for img in a:
        label_im = label(img)
        imgs = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
        imgs = cv2.resize(imgs,(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(imgs),np.array(label_im)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
b=[]
def create_test_data():
    testing_data = []
    for f in os.listdir(i):
	      b.append(os.path.join(i,f))
    for img in b:
        img_num = img.split('.')[0]
        img_data = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
train_data = create_train_data()
test_data = create_test_data()
train = train_data
test = test_data
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
#model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10, 
 #         validation_set=({'input': X_test}, {'targets': y_test}), 
  #       snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
print(test_data)
d = test_data[0]
img_data, img_num = d

data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
prediction = model.predict([data])[0]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.imshow(img_data, cmap="gray")
print(f"baby: {prediction[0]}, doll: {prediction[1]}")
