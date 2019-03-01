
# coding: utf-8

# *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Python Machine Learning - Code Examples

# # Chapter 12 - Implementing a Multi-layer Artificial Neural Network from Scratch
# 


# ## Obtaining the MNIST dataset

# The MNIST dataset is publicly available at http://yann.lecun.com/exdb/mnist/ and consists of the following four parts:
# 
# - Training set images: train-images-idx3-ubyte.gz (9.9 MB, 47 MB unzipped, 60,000 samples)
# - Training set labels: train-labels-idx1-ubyte.gz (29 KB, 60 KB unzipped, 60,000 labels)
# - Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 7.8 MB, 10,000 samples)
# - Test set labels: t10k-labels-idx1-ubyte.gz (5 KB, 10 KB unzipped, 10,000 labels)
# 
# In this section, we will only be working with a subset of MNIST, thus, we only need to download the training set images and training set labels. After downloading the files, I recommend unzipping the files using the Unix/Linux gzip tool from the terminal for efficiency, e.g., using the command 
# 
#     gzip *ubyte.gz -d
#  
# in your local MNIST download directory, or, using your favorite unzipping tool if you are working with a machine running on Microsoft Windows. The images are stored in byte form, and using the following function, we will read them into NumPy arrays that we will use to train our MLP.
# 


import os
import struct
import numpy as np
 
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, 
                               '%s-images-idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
 
    return images, labels



# unzips mnist

import sys
import gzip
import shutil

if (sys.version_info > (3, 0)):
    writemode = 'wb'
else:
    writemode = 'w'

zipped_mnist = [f for f in os.listdir('./') if f.endswith('ubyte.gz')]
for z in zipped_mnist:
    with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
        outfile.write(decompressed.read()) 


# Load training records
X_train, y_train = load_mnist('', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))


# Load testing records
X_test, y_test = load_mnist('', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

'''
# Visualize the first digit of each class:

import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.savefig('images/12_5.png', dpi=300)
plt.show()



# Visualize 25 different versions of "7":

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.savefig('images/12_6.png', dpi=300)
plt.show()

'''

# ## Implementing a multi-layer perceptron
##made changes to the initial code

#implementing sklearn MLPClassifier
from sklearn.neural_network import MLPClassifier
import sys

n_epochs = 200

#get arguments

#default values of parameter
num_neurons_value = 100
activation_value = 'logistic'
batch_size_value = 100
learning_rate_value = 0.0005

#parameter value from command line
paramerter_type = sys.argv[2]
print(sys.argv[1],'+',paramerter_type)

#upadating required parameter
if(paramerter_type == 'nn') : num_neurons_value = int(sys.argv[1])
elif(paramerter_type == 'av') :	activation_value = sys.argv[1]
elif(paramerter_type == 'bs') :	batch_size_value = int(sys.argv[1])
elif(paramerter_type == 'lr') :	learning_rate_value = int(sys.argv[1])

print(learning_rate_value,
	batch_size_value,
	activation_value,
	num_neurons_value)
#paramters to change
  #hidden_layer_sizes = (100,100) for 2 layer neuron each with 100 neurons
  #number of neurons ^^^^^
  #activation -->tanh
  #batch_size, update weight after each 
  #learning_rate_init
  #nn = MLPClassifier(hidden_layer_sizes = (int(arg_script),),

nn = MLPClassifier(hidden_layer_sizes = (170,num_neurons_value), 
	activation = activation_value, batch_size = batch_size_value,
	max_iter = n_epochs, learning_rate = 'constant', 
	learning_rate_init = learning_rate_value) 


nn.fit(X_train[:55000], 
       y_train[:55000])


# ---
# **Note**
# 
# In the fit method of the MLP example above,
# 
# ```python
# 
# for idx in mini:
# ...
#     # compute gradient via backpropagation
#     grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
#                                       a3=a3, z2=z2,
#                                       y_enc=y_enc[:, idx],
#                                       w1=self.w1,
#                                       w2=self.w2)
# 
#     delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
#     self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
#     self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
#     delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
# ```
# 
# `delta_w1_prev` (same applies to `delta_w2_prev`) is a memory view on `delta_w1` via  
# 
# ```python
# delta_w1_prev = delta_w1
# ```
# on the last line. This could be problematic, since updating `delta_w1 = self.eta * grad1` would change `delta_w1_prev` as well when we iterate over the for loop. Note that this is not the case here, because we assign a new array to `delta_w1` in each iteration -- the gradient array times the learning rate:
# 
# ```python
# delta_w1 = self.eta * grad1
# ```
# 
# The assignment shown above leaves the `delta_w1_prev` pointing to the "old" `delta_w1` array. To illustrates this with a simple snippet, consider the following example:
# 
# 
'''
import matplotlib.pyplot as plt

plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
#plt.savefig('images/12_07.png', dpi=300)
plt.show()


# In[22]:


plt.plot(range(nn.epochs), nn.eval_['train_acc'], 
         label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'], 
         label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
#plt.savefig('images/12_08.png', dpi=300)
plt.show()
'''

# In[23]:


#y_test_pred = nn.predict(X_train[55000:])
#acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])

acc = nn.score(X_test, y_test)

print('Test accuracy: %.2f%%' % (acc * 100))

#write data to the file
data = 'Test accuracy: ' + str(acc * 100)+'% \n'+ str(learning_rate_value)+' '+str(batch_size_value)+' '+activation_value+' '+str(num_neurons_value)+' 1 \n\n'
with open('data.txt', 'a') as file:
  file.write(data)

'''
# In[24]:


miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
#plt.savefig('images/12_09.png', dpi=300)
plt.show()
'''