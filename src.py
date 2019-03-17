from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import matplotlib.pyplot as plt

import functions as f


fscore1=[]
fscore2=[]
fscore3=[]
fscore4=[]
filename1 = 'wine.data.csv'
filename2 = 'diabetes.data.csv'
filename3 = 'glass.data.csv'
filename4 = 'abalone_improved.data.csv'
dataset1 = f.import_file(filename1)
dataset2 = f.import_file(filename2)
dataset3 = f.import_file(filename3)
dataset4 = f.import_file(filename4)


data1, target1=f.break_class_wine(dataset1)
data2, target2=f.break_class_other(dataset2)
data3, target3=f.break_class_other(dataset3)
data4, target4=f.break_class_other(dataset4)
data1 = np.array(data1)
'''for i in range(0, 5):
    for j in range (0, 13):
        k = np.float(data1[i][j])
        print(k)
    print("\n")'''
data2 = np.array(data2)
data3 = np.array(data3)
data4 = np.array(data4)
data1 = preprocessing.normalize(data1, axis=0, norm='l2')
'''for i in range(0, 5):
    for j in range (0, 13):
        k = np.float(data1[i][j])
        print("%.2f" % k)
    print("\n")'''
data2 = preprocessing.normalize(data2, axis=0, norm='l2')
data3 = preprocessing.normalize(data3, axis=0, norm='l2')
data4 = preprocessing.normalize(data4, axis=0, norm='l2')
target1=np.array(target1)
target2=np.array(target2)
target3=np.array(target3)
target4=np.array(target4)

'''
------------------------------------BADANIA A---------------------------------------------
'''
'''

# for i in range (2,11):
for i in range (2,50):
    skf1 = StratifiedKFold(n_splits=10)
    for train_index1, test_index1 in (skf1.split(data1,target1)):
        X_train1 = data1[train_index1]
        X_test1 = data1[test_index1]
        y_train1, y_test1 = target1[train_index1], target1[test_index1]

    skf2 = StratifiedKFold(n_splits=6)
    for train_index2, test_index2 in (skf2.split(data2, target2)):
        X_train2 = data2[train_index2]
        X_test2 = data2[test_index2]
        y_train2, y_test2 = target2[train_index2], target2[test_index2]

    skf3 = StratifiedKFold(n_splits=2)
    for train_index3, test_index3 in (skf3.split(data3, target3)):
        X_train3 = data3[train_index3]
        X_test3 = data3[test_index3]
        y_train3, y_test3 = target3[train_index3], target3[test_index3]

    skf4 = StratifiedKFold(n_splits=5)
    for train_index4, test_index4 in (skf4.split(data4, target4)):
        X_train4 = data4[train_index4]
        X_test4 = data4[test_index4]
        y_train4, y_test4 = target4[train_index4], target4[test_index4]


    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train1,y_train1)
    predicted=neigh.predict(X_test1)
    fscore1.append(f1_score(y_test1, predicted, average='micro'))

    neigh.fit(X_train2, y_train2)
    predicted = neigh.predict(X_test2)
    fscore2.append(f1_score(y_test2, predicted, average='micro'))

    neigh.fit(X_train3, y_train3)
    predicted = neigh.predict(X_test3)
    fscore3.append(f1_score(y_test3, predicted, average='micro'))

    neigh.fit(X_train4, y_train4)
    predicted = neigh.predict(X_test4)
    fscore4.append(f1_score(y_test4, predicted, average='micro'))


print(fscore1)
print(fscore2)
print(fscore3)
print(fscore4)


def max(list):
    max=0
    index=0
    for i in range(len(list)):
        if list[i]>max:
            max=list[i]
            index=i+1
    return max, index

f1max, index1 = max(fscore1)
f2max, index2 = max(fscore2)
f3max, index3 = max(fscore3)
f4max, index4 = max(fscore4)


print(f1max, index1)
print(f2max, index2)
print(f3max, index3)
print(f4max, index4)


# plotowanie wykresów

plt.figure(1)
# X=np.linspace(2,13,9)
X=np.arange(2, 50)


plt.title('Amount of nearest neighbours')
plt.xlabel('N.N.')
plt.ylabel('F-score')
plt.plot(X, fscore1, 'o--', label="Wines")
plt.plot(X, fscore2, 'o--', label="Diabetes")
plt.plot(X, fscore3, 'o--', label="Glass")
plt.plot(X, fscore4, 'o--', label="Abalone")
plt.legend()
plt.show()
'''



'''
------------------------------------BADANIA B---------------------------------------------
'''



skf1 = StratifiedKFold(n_splits=10)
for train_index1, test_index1 in (skf1.split(data1,target1)):
    X_train1 = data1[train_index1]
    X_test1 = data1[test_index1]
    y_train1, y_test1 = target1[train_index1], target1[test_index1]

skf2 = StratifiedKFold(n_splits=6)
for train_index2, test_index2 in (skf2.split(data2, target2)):
    X_train2 = data2[train_index2]
    X_test2 = data2[test_index2]
    y_train2, y_test2 = target2[train_index2], target2[test_index2]

skf3 = StratifiedKFold(n_splits=2)
for train_index3, test_index3 in (skf3.split(data3, target3)):
    X_train3 = data3[train_index3]
    X_test3 = data3[test_index3]
    y_train3, y_test3 = target3[train_index3], target3[test_index3]

skf4 = StratifiedKFold(n_splits=5)
for train_index4, test_index4 in (skf4.split(data4, target4)):
    X_train4 = data4[train_index4]
    X_test4 = data4[test_index4]
    y_train4, y_test4 = target4[train_index4], target4[test_index4]


neigh1 = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2, weights=f.squared_distance)
neigh1.fit(X_train1,y_train1)
predicted=neigh1.predict(X_test1)
fscore1 = (f1_score(y_test1, predicted, average='micro'))

neigh2 = KNeighborsClassifier(n_neighbors=4, metric='minkowski', p=2, weights=f.squared_distance)
neigh2.fit(X_train2, y_train2)
predicted = neigh2.predict(X_test2)
fscore2 = (f1_score(y_test2, predicted, average='micro'))

neigh3 = KNeighborsClassifier(n_neighbors=24, metric='minkowski', p=2, weights=f.squared_distance)
neigh3.fit(X_train3, y_train3)
predicted = neigh3.predict(X_test3)
fscore3 = (f1_score(y_test3, predicted, average='micro'))

neigh4 = KNeighborsClassifier(n_neighbors=32, metric='minkowski', p=2, weights=f.squared_distance)
neigh4.fit(X_train4, y_train4)
predicted = neigh4.predict(X_test4)
fscore4 = (f1_score(y_test4, predicted, average='micro'))

print("%.4f" % fscore1)
print("%.4f" % fscore2)
print("%.4f" % fscore3)
print("%.4f" % fscore4)

