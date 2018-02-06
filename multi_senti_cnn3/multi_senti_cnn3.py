
import numpy
import re
import os.path
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
import NormalizeCorpusTest
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, average_precision_score
from keras.callbacks import ModelCheckpoint


def getPickledData(pickle_file):
    # load from file
    f = open(pickle_file, "r")
    (X_train, y_train), (X_dev, y_dev), (X_test1, y_test1), (X_test2, y_test2), top_words = pickle.load(f)
    f.close()
    return (X_train, y_train), (X_dev, y_dev), (X_test1, y_test1), (X_test2, y_test2), top_words


def writePickleData(pickle_file, p_list):
    # write a file
    f = open(pickle_file, "w")
    pickle.dump(p_list, f)
    f.close()


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
pickle_file = "data/corpus_germeval_pickled_sent"
train_file = "data/sentiment/train.tsv"
dev_file = "data/sentiment/dev.tsv"
test_file1 = "data/test_answers/test_TIMESTAMP1-sem.tsv"
test_file2 = "data/test_answers/test_TIMESTAMP2-sem.tsv"

if os.path.isfile(pickle_file):
    (X_train, y_train), (X_dev, y_dev), (X_test1, y_test1), (X_test2, y_test2), top_words = getPickledData(pickle_file)
else:
    (X_train, y_train), (X_dev, y_dev), (X_test1, y_test1), (X_test2, y_test2), top_words = NormalizeCorpusTest.semDBNormalize([train_file, dev_file, test_file1, test_file2])
    writePickleData(pickle_file, [(X_train, y_train), (X_dev, y_dev), (X_test1, y_test1), (X_test2, y_test2), top_words])

label_num = len(set(y_train))

max_words = 300
filter_size = 300
net_size = 600
epochs = 2
# pad dataset to a maximum review length in words
X_train = sequence.pad_sequences(X_train, maxlen=max_words, padding='post')
X_dev = sequence.pad_sequences(X_dev, maxlen=max_words, padding='post')
X_test1 = sequence.pad_sequences(X_test1, maxlen=max_words, padding='post')
X_test2 = sequence.pad_sequences(X_test2, maxlen=max_words, padding='post')

y_dev_gold = y_dev
y_test1_gold = y_test1
y_test2_gold = y_test2
y_train = np_utils.to_categorical(y_train)
y_dev = np_utils.to_categorical(y_dev)
y_test1 = np_utils.to_categorical(y_test1)
y_test2 = np_utils.to_categorical(y_test2)
y_train = numpy.array(y_train)
y_dev = numpy.array(y_dev)
y_test1 = numpy.array(y_test1)
y_test2 = numpy.array(y_test2)

print('Building the NN model...')
model = Sequential()
model.add(Embedding(input_dim=top_words, output_dim=300, input_length=max_words))
model.add(Conv1D(activation="relu", padding="valid", filters=filter_size, kernel_size=7))
model.add(GlobalMaxPooling1D())
model.add(Dense(600, activation='relu'))
model.add(Dense(label_num, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#checkpoint
filepath="weights-multi-best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor= 'val_acc' , verbose=1, save_best_only=True,
mode= 'max' )
callbacks_list = [checkpoint]

model.fit(X_train, y_train, batch_size=20, epochs=epochs, validation_data=(X_dev, y_dev), callbacks=callbacks_list)

#load weights and recompile
model.load_weights(filepath)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# evaluate the model
print "-" * 10
print model.layers
print "-" * 10
print "evaluation on dev"
scores = model.evaluate(X_dev, y_dev)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
y_pred = model.predict(X_dev)
y_pred = y_pred.argmax(1)
print confusion_matrix(y_dev_gold, y_pred)
print "Accuracy Rate by 'accuracy_score' is: %f" % accuracy_score(y_dev_gold, y_pred)
print "Accuracy Rate by 'f1_score macro' is: %f" % f1_score(y_dev_gold, y_pred, average='macro')


print "evaluation on test1"
scores = model.evaluate(X_test1, y_test1)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
y_test1_pred = model.predict(X_test1)
y_test1_pred = y_test1_pred.argmax(1)
print confusion_matrix(y_test1_gold, y_test1_pred)
print "Accuracy Rate by 'accuracy_score' is: %f" % accuracy_score(y_test1_gold, y_test1_pred)
print "Accuracy Rate by 'f1_score macro' is: %f" % f1_score(y_test1_gold, y_test1_pred, average='macro')

print "evaluation on test2"
scores = model.evaluate(X_test2, y_test2)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
y_test2_pred = model.predict(X_test2)
y_test2_pred = y_test2_pred.argmax(1)
print confusion_matrix(y_test2_gold, y_test2_pred)
print "Accuracy Rate by 'accuracy_score' is: %f" % accuracy_score(y_test2_gold, y_test2_pred)
print "Accuracy Rate by 'f1_score macro' is: %f" % f1_score(y_test2_gold, y_test2_pred, average='macro')


