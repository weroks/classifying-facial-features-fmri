from sklearn import svm
from sklearn.metrics import accuracy_score

train_samples = np.load('train_samples.npy')
train_targets = np.load('train_targets.npy', allow_pickle=True)
test_samples = np.load('test_samples.npy')
test_targets = np.load('test_targets.npy', allow_pickle=True)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernels:
    print('\n kernel: ', k)
    clf = svm.SVC(kernel=k)
    clf.fit(train_samples, train_targets)
    print('model fitted')
    train_pred = clf.predict(train_samples)
    train_acc = accuracy_score(train_targets, train_pred)
    print('training accuracy: ', train_acc)
    test_pred = clf.predict(test_samples)
    test_acc = accuracy_score(test_targets, test_pred)
    print('testing accuracy: ', test_acc)
