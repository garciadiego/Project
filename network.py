#libraries 
import matplotlib.pyplot as plt
import tarfile
import numpy as np
from numpy import random
import os, sys
import tensorflow as tf
from scipy import ndimage
import cPickle as pickle
import math


#Hyperparameters
num_steps = 5001
batch_size = 500
beta = 0.0001  #L2 regularization
learning_rate = 0.5
dropout_rate = 0.95 #dropout 50% percentage
adam_learning_rate = 1e-4

#Hidden Units
hidden_units1 = 500
hidden_units2 = 500

#Data Base Management
num_train_images = 30000
num_test_images = 1200
image_size = 28   # 28x28
pixel_depth = 255.0  # Number of levels per pixel.

#Training, Validation and test Length
train_size = 30000  #size of the training data
valid_size = 1200   #size of the validation data
test_size = 1200    #size of the test data

#System Setup
num_labels = 21
num_classes = 21
display_step = 100



#### Extracts the Files if no present #####
def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

####Prepares Data for Training (Labeling, Stad Dev, etc) and save it##########

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  for image_index, image in enumerate(image_files):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) -
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :] = image_data
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

  num_images = image_index + 1
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))

  print('Tensor Shape:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)

  return dataset_names

train_datasets = maybe_pickle(train_folders, num_train_images )
test_datasets = maybe_pickle(test_folders, num_test_images)

############ Dataset Preprocess ##############

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes

  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # shuffle the letters to have random validation and training set
        np.random.seed(1) #Reset Random State
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class

        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise

  return valid_dataset, valid_labels, train_dataset, train_labels


valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)


######Shows a sample from the pickle file#######
plt.imshow(train_dataset[2000]) # Test image from the ndarray, to verify data!
plt.show()

#Randomize the data
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

#I am saving all training, validation and test data
pickle_file = 'all_data.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

pickle_file = 'all_data.pickle'

######################Neural Network################################


#####Reformat the datasets for the Deep Network######
def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # One Hot Encoding ex. Map 0 to [1, 0, 0 ...], 1 to [0, 1, 0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

################### Deep Neural Network #############################

graph = tf.Graph()
with graph.as_default():
  #Input and Output Place Holders
  X = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
  Y = tf.placeholder(tf.float32,shape=(batch_size, num_labels))
  #Test and Valid Data set
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Layer 1.
  w_1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_units1],
                                              stddev=math.sqrt(2.0/(image_size*image_size)),
                                              name="weight1"))
  b_1 = tf.Variable(tf.zeros([hidden_units1],name="Bias1"))
  h_1 = tf.nn.relu(tf.matmul(X, w_1) + b_1)
  h_1 = tf.nn.dropout(h_1, dropout_rate)

  # Layer 2
  w_2 = tf.Variable(tf.truncated_normal([hidden_units1, hidden_units2],
                                             stddev=math.sqrt(2.0/(hidden_units1)),
                                             name="weight2"))
  b_2 = tf.Variable(tf.zeros([hidden_units2],name="Bias2"))
  h_2 = tf.nn.relu(tf.matmul(h_1, w_2) + b_2)
  h_2 = tf.nn.dropout(h_2, dropout_rate)

  # Output
  w_3 = tf.Variable(tf.truncated_normal([hidden_units2, num_labels],
                                            stddev=math.sqrt(2.0/(hidden_units2)),
                                            name="weight3"))
  b_3 = tf.Variable(tf.zeros([num_labels], name="Bias3"))
  output = tf.matmul(h_2, w_3) + b_3
  train_output = tf.nn.softmax(output)
  train_output = tf.nn.dropout(output, dropout_rate)

  #Loss function with L2 regularization
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(output, Y)
    + (beta*(tf.nn.l2_loss(w_1) + tf.nn.l2_loss(w_2) + tf.nn.l2_loss(w_3))))


#########################  OPTIMIZER  ####################################
  # Gradient Descend
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  #Adam Optimizer
  #optimizer = tf.train.AdamOptimizer(adam_learning_rate).minimize(loss)
#############################################################################

  # Predictions for the training, validation, and test data.
  vp_h1 = tf.nn.relu(tf.matmul(tf_valid_dataset, w_1) + b_1)
  vp_h2 = tf.nn.relu(tf.matmul(vp_h1, w_2) + b_2)
  valid_prediction = tf.nn.softmax(tf.matmul(vp_h2, w_3) + b_3)

  test_h1 = tf.nn.relu(tf.matmul(tf_test_dataset, w_1) + b_1)
  test_h2 = tf.nn.relu(tf.matmul(test_h1, w_2) + b_2)
  test_prediction = tf.nn.softmax(tf.matmul(test_h2, w_3) + b_3)

######## Tensorboard #########

  #Tensorboard Accuracy
  correct_accuracy = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
  test_accuracy = tf.reduce_mean(tf.cast(correct_accuracy, tf.float32))

  #Tensorboard Histograms
  tf.histogram_summary('weights 1', w_1)
  tf.histogram_summary('weights 2', w_2)
  tf.histogram_summary('weights 3', w_3)
  tf.histogram_summary('Output', output)

  #Tensorboard Summary
  tf.scalar_summary("Loss", loss)
  tf.scalar_summary('Accuracy', test_accuracy)
  merged = tf.merge_all_summaries ()


####### Run the Network and Displays results #########

with tf.Session(graph=graph) as session:
#Tensorboard
  tf.initialize_all_variables().run()
  saver = tf.train.Saver()
  writer = tf.train.SummaryWriter("/home/diego/Desktop/Project/code/Tensorboard/",
                                                     graph=tf.get_default_graph())

  print("############  DEEP NEURAL NETWORK IS RUNNING....  ############")

  for step in range(num_steps):
    
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {X : batch_data, Y : batch_labels}
    _, l, predictions,summary = session.run([optimizer, loss, train_output, merged],
                                                    feed_dict=feed_dict)
#Tensorboard Writter and summary
    writer.add_summary(summary, step)
#Save the model to use later
    save_path = saver.save(session, "/home/diego/Desktop/Project/code/tf_saver/model.ckpt")

#Accuracy Results
    if (step % display_step == 0):
      print("Step %d completed out of %d loss: %f" % (step, num_steps, l))
      print("Batch Accuracy: %.2f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.2f%%" % accuracy(
                                        valid_prediction.eval(), valid_labels))

  print('####### Network Optimization Finish #######')
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
  print("Model saved in file: %s" % save_path)



with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  
  print("####### SCIKIT LEARN CLASSIFIERS RESULT##########")
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)



#########OTHER CLASSIFIERS TEST######

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
lrm = LogisticRegression(penalty='l2')
data_len = 5000
Ytrain = train_labels[:data_len]
Xtrain = train_dataset[:data_len]
X2dim = Xtrain.reshape(len(Xtrain),-1)
ytest = train_labels[data_len+1:2*data_len]
Xtest = train_dataset[data_len+1:2*data_len]
lrm.fit(X2dim, Ytrain)
ypred= lrm.predict(Xtest.reshape(len(Xtest),-1))
#print('Logistic Regression Results: ')
#print(accuracy_score(ytest, ypred))
#print(confusion_matrix(ytest, ypred))

#Support Vector Classifier
from sklearn.svm import SVC
#svm = SVC(kernel='linear')
svm = SVC(gamma=0.001, C=100)
svm.fit(X2dim,Ytrain)
ypred= svm.predict(Xtest.reshape(len(Xtest),-1))
print('Support Vector Classifier Result: ')
print(accuracy_score(ytest, ypred))
#print(confusion_matrix(ytest, ypred))

#Does not work yet
#print('Prediction:', svm.predict(X2dim[-2]))
#plt.imshow(train_dataset[-2], cmap=plt.cm.gray_r, interpolation="nearest")
#plt.show()

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=999)
clf.fit(X2dim, Ytrain)
ypred = clf.predict(Xtest.reshape(len(Xtest),-1))
print('Random Forest Classifier Result: ')
print(accuracy_score(ytest, ypred))
#print(confusion_matrix(ytest, ypred))
