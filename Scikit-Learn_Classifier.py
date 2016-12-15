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
