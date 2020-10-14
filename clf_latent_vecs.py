import numpy as np
import h5py
from make_network import *
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from scipy.stats import binom_test

# load the trained vaegan
sess, X, G, Z, Z_mu, is_training, saver = make_network()
saver.restore(sess, "vaegan_celeba.ckpt")

def latent2im(latent):
    '''
    Creates an image from a latent vector.
    This method is taken from the following work by VanRullen and Reddy (2019):
    https://www.nature.com/articles/s42003-019-0438-y
    Small adjustments have been made.
    '''
    if latent.shape[0] == 1024:
        latent = latent[np.newaxis]

    #generate images from z
    g = sess.run(G, feed_dict={Z: (1*latent), is_training: False})

    def imdeprocess(g):
        stretch = 1.0
        for i in range(g.shape[0]):
            g[i]=np.clip(stretch*g[i] / (g.max()),0,1)
        return g

    g = imdeprocess(g)
    return g[0]

def brain2latent(activ_pattern, W, invcovW):
    '''
    Calculates the 1024 dimensional latent vector corresponding to a given voxel activation pattern.
    This method is taken from the following work by VanRullen and Reddy (2019):
    https://www.nature.com/articles/s42003-019-0438-y
    Small adjustments have been made.
    '''
    # zero all nans in activation pattern
    activ_pattern[np.isnan(activ_pattern)] = 0
    # decode by multiplying with learned weight matrix W and its inverse covariance matrix
    latent_code = activ_pattern @ W.T @ invcovW
    # remove bias term at the end of the vector
    return latent_code[:-1]

def load_data(sub):
    # select folder
    folder = 'derivatives/sub-0' + str(sub)

    # load voxel activation patterns from viewing of test images
    file_name = Path(folder + '/test_patterns.mat')
    patterns = h5py.File(file_name,'r')
    # convert to numpy array for easier handling and transpose due to file loading error
    patterns = np.array(patterns.get('vol')).T

    # load indices of relevant voxels with corresponding W
    roi_file_name = Path(folder + '/ROI_VAEGAN.mat')
    roi = h5py.File(roi_file_name,'r')
    W = np.array(roi.get('W')).T
    invcovW = np.array(roi.get('invcovW')).T
    val_idxs = np.array(roi.get('validindices'))[0].astype(int)

    # filter activation patterns by indexing with valid indices
    fmri_vecs = [p[val_idxs] for p in patterns]

    # load latent vectors of test images
    ims_file_name = Path(folder + '/test_images.mat')
    test_ims_l = h5py.File(ims_file_name,'r')
    # convert to numpy array for easier handling and transpose due to file loading error
    test_ims_l_vecs = np.array(test_ims_l.get('testimage_vaeganlatentvars')).T

    return fmri_vecs, test_ims_l_vecs, W, invcovW

def euclidean_clf(l_vecs, axis_v, attrib):
    '''
    Performs euclidean classification on latent vectors
    based on sign of projection onto the given attribute axis.

    Args:
        l_vecs (list): List of latent vectors to be classified.
        class1 (array): Array of latent vectors belonging to class 1 (eg. male).
        class2 (array): Array of latent vectors belonging to class 2 (eg. female).
        attrib (str): Either 'gender' or 'expression'.

    Returns:
        labels (numpy array): Array of resulting labels.
    '''

    norm = np.linalg.norm(axis_v)
    # project each vector onto 'male direction' of gender axis
    proj = np.array([np.dot(vec, axis_v)/norm for vec in l_vecs])
    if attrib == 'gender':
        # assign proper label depending on sign of projection
        labels = ['Male' if p > 0 else 'Female' for p in proj]
    else:
        labels = ['Smiling' if p > 0 else 'Neutral' for p in proj]

    return np.array(labels)

def create_dataset(class1, class2):
    data = np.concatenate((class1, class2)).reshape(1000,1024)
    # create target array
    targets = np.zeros(1000)
    targets[:500] = 1
    return data, targets

fmri_vecs = [load_data(i+1)[0] for i in range(4)]
orig_vecs = [load_data(i+1)[1] for i in range(4)]
Ws = [load_data(i+1)[2] for i in range(4)]
invcovWs = [load_data(i+1)[3] for i in range(4)]

# create list of decoded latent vectors
dec_vecs = [[brain2latent(vec, Ws[i], invcovWs[i]) for vec in fmri_vecs[i]] for i in range(4)]

# feed decoded latent vectors into generator
dec_faces = [[latent2im(vec) for vec in sub] for sub in dec_vecs]

true_gender = ['Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female']
true_exp = [np.load(Path('true_exp/true_exp_s' + str(i+1) + '.npy')) for i in range(4)]

male_vecs = np.load(Path('vecs/male_vecs.npy'))
female_vecs = np.load(Path('vecs/female_vecs.npy'))
smiling_vecs = np.load(Path('vecs/smiling_vecs.npy'))
neutral_vecs = np.load(Path('vecs/neutral_vecs.npy'))

# compute vector pointing from mean_class2 to mean_class1
axis_g = (np.mean(male_vecs, axis=0) - np.mean(female_vecs, axis=0))[0]
axis_e = (np.mean(smiling_vecs, axis=0) - np.mean(neutral_vecs, axis=0))[0]

### Euclidean Classification ###
gen_acc_o, gen_acc_d = [], []
exp_acc_o, exp_acc_d = [], []
for i in range(4):
    # for each subject...
    # perform gender classification
    g_labels_o = euclidean_clf(orig_vecs[i], axis_v=axis_g, attrib='gender')
    g_labels_d = euclidean_clf(dec_vecs[i], axis_v=axis_g, attrib='gender')
    # compute accuracy
    gen_acc_o.append(accuracy_score(true_gender, g_labels_o))
    gen_acc_d.append(accuracy_score(true_gender, g_labels_d))
    # perform expression classification
    e_labels_o = euclidean_clf(orig_vecs[i], axis_v=axis_e, attrib='expression')
    e_labels_d = euclidean_clf(dec_vecs[i], axis_v=axis_e, attrib='expression')
    # compute accuracy
    exp_acc_o.append(accuracy_score(true_exp[i], e_labels_o))
    exp_acc_d.append(accuracy_score(true_exp[i], e_labels_d))

print('\n Euclidean Classification Accuracies\n')
print('Gender: ')
print('\t Original Vectors: ', np.mean(gen_acc_o))
print('\t Decoded Vectors: ', np.mean(gen_acc_d))
print('Expression: ')
print('\t Original Vectors: ', np.mean(exp_acc_o))
print('\t Decoded Vectors: ', np.mean(exp_acc_d))

def nested_cv(data, targets):
    '''
    Nested Cross-Validation.
    This method is taken from Jason Brownlees Tutorial:
    https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
    '''
    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(data):
        # split data
        X_train, X_test = data[train_ix, :], data[test_ix, :]
        y_train, y_test = targets[train_ix], targets[test_ix]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        # define the model
        model = SVC(random_state=1)
        # define search space
        space = dict()
        space['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
        space['C'] = [0.5, 0.75, 1, 4]
        # define search
        search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        # store the result
        outer_results.append(acc)
        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
    # summarize the estimated performance of the model
    print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))
    return


### SVM Classification ###

g_data = create_dataset(male_vecs, female_vecs)
# hyperparameter tuning using nested cross validation
# nested_cv(g_data[0], g_data[1])
# use hyperparameter with highest accuracy in the training set (look at output from nested cv)
g_clf = SVC(C=1, kernel='sigmoid')
g_clf.fit(g_data[0], g_data[1])

svm_g_orig_acc, svm_g_dec_acc = [], []

for i in range(4):
    g_orig_svm_l = g_clf.predict(orig_vecs[i])
    g_dec_svm_l = g_clf.predict(dec_vecs[i])
    g_orig_l = ['Male' if g == 1 else 'Female' for g in g_orig_svm_l]
    g_dec_l = ['Male' if g == 1 else 'Female' for g in g_dec_svm_l]

    svm_g_orig_acc.append(accuracy_score(true_gender, g_orig_l))
    svm_g_dec_acc.append(accuracy_score(true_gender, g_dec_l))

print('\n SVM Classification Accuracies \n')
print('Gender: ')
print('\t Original Vectors: ', np.mean(svm_g_orig_acc))
print('\t Decoded Vectors: ', np.mean(svm_g_dec_acc))

e_data = create_dataset(smiling_vecs, neutral_vecs)
#nested_cv(e_data[0], e_data[1])
e_clf = SVC(C=0.75, kernel='sigmoid')
e_clf.fit(e_data[0], e_data[1])

svm_e_orig_acc, svm_e_dec_acc = [], []

for i in range(4):
    e_orig_svm_l = e_clf.predict(orig_vecs[i])
    e_dec_svm_l = e_clf.predict(dec_vecs[i])
    e_orig_l = ['Smiling' if g == 1 else 'Neutral' for g in e_orig_svm_l]
    e_dec_l = ['Smiling' if g == 1 else 'Neutral' for g in e_dec_svm_l]

    svm_e_orig_acc.append(accuracy_score(true_exp[i], e_orig_l))
    svm_e_dec_acc.append(accuracy_score(true_exp[i], e_dec_l))

print('Expression: ')
print('\t Original Vectors: ', np.mean(svm_e_orig_acc))
print('\t Decoded Vectors: ', np.mean(svm_e_dec_acc))

# check for statistical significance
accs = [gen_acc_d, exp_acc_d, svm_g_dec_acc, svm_e_dec_acc]
mean_accs = [np.mean(ac) for ac in accs]
correct_samples = [ac * 80 for ac in mean_accs]
print('\n p-values for classification of decoded vectors: ')
for s in correct_samples:
    print(binom_test(s, 80, 0.5))

'''
OUTPUT:

 Euclidean Classification Accuracies

Gender:
         Original Vectors:  0.9125000000000001
         Decoded Vectors:  0.7
Expression:
         Original Vectors:  0.8374999999999999
         Decoded Vectors:  0.725

 SVM Classification Accuracies

Gender:
         Original Vectors:  0.9
         Decoded Vectors:  0.75
Expression:
         Original Vectors:  0.8125
         Decoded Vectors:  0.7375

 p-values for classification of decoded vectors:
0.0004515172719148108
7.010573979919962e-05
8.58055986704962e-06
2.529110256467062e-05
'''
