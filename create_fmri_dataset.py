import numpy as np
import pandas as pd
import os
from mvpa2.datasets.mri import fmri_dataset

cwd = os.getcwd()
datapath = os.path.join(cwd, 'preproc_data')

# load mask image
mask_fname = os.path.join(datapath, 'mask.nii.gz')

def load_events_to_df(ses):
    # get path to directory of current session
    ses_dir = os.path.join(datapath, 'ses-0' + str(ses), 'events')
    cols = ['onset', 'trial_type', 'stim_file']
    # get list with one dataframe per run
    df_list = [pd.read_csv(ses_dir + '/' + f, sep='\t', usecols=cols) for f in os.listdir(ses_dir)]
    for df in df_list:
        df.columns = ['onset', 'trial_type', 'expression']
    return df_list

def load_bold_ds(ses):
    # get path to directory of current session
    ses_dir = os.path.join(datapath, 'ses-0' + str(ses))
    # convert to fmri_dataset format
    d_sets = []
    for f_name in os.listdir(ses_dir):
        f = os.path.join(ses_dir, f_name)
        if os.path.isfile(f):
            d_sets.append(fmri_dataset(f, mask = mask_fname))
    return d_sets

def create_dataset(ses, trial_type):
    dfs = load_events_to_df(ses)
    dss = load_bold_ds(ses)
    # collection of samples and targets for all runs
    X, Y = [], []

    trials = [df.loc[df['trial_type'] == trial_type] for df in dfs]
    # iterate through all runs of current session
    for i in range(len(trials)):
        # get event dataframe and bold dataset corresponding to run i
        df, ds = trials[i], dss[i]
        # get list of expression labels as targets
        targets = np.array(df['expression'])
        # get timepoints corresponding to when stimulus was shown (in scans)
        onsets = np.array(df['onset'])
        # bold response is computed by calculating the mean between volumes acquiered
        # 4, 6 and 8 seconds after stimulus onset
        samples = np.array([np.mean([ds[int((t+4)/2)].samples[0], ds[int((t+6)/2)].samples[0],
                                    ds[int((t+8)/2)].samples[0]], axis=0) for t in onsets])
        samples = np.float32(samples)
        X.append(samples)
        Y.append(targets)

    return np.concatenate(X), np.concatenate(Y)


train_samples, train_targets = [], []
test_samples, test_targets = [], []
sessions = np.arange(8) + 1

for s in sessions:
    print('working on session ', s, '...')
    x, y = create_dataset(s, 'train_face')
    train_samples.append(x)
    train_targets.append(y)
    x, y = create_dataset(s, 'test_face')
    test_samples.append(x)
    test_targets.append(y)
    print('done.')

train_samples = np.vstack(train_samples)
train_targets = np.hstack(train_targets)
test_samples = np.vstack(test_samples[2:])
test_targets = np.hstack(test_targets[2:])

# delete comlumns with zero variance
n_feats = train_samples.shape[1]
useless_cols = [i if (np.var(train_samples[:,i]) == 0) for i in range(n_feats)]

train_samples = np.delete(train_samples, useless_cols, axis=1)
test_samples = np.delete(test_samples, useless_cols, axis=1)

# save formatted data
#np.save('train_samples', train_samples)
#np.save('train_targets', train_targets)
#np.save('test_samples', test_samples)
#np.save('test_targets', test_targets)
