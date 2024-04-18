'''
Interface helpers
'''

import numpy as np

def read_in_preds(fpath, modelname, q_upper, q_lower, readmean, readstd, readwts):
    """
    Private function, read in predictions from text files.
    """
    mdraws = []
    sdraws = []
    wdraws = []
    if readmean:
        mdraw_files = sorted(
            list(fpath.glob(modelname + ".mdraws*")))
        
        for f in mdraw_files:
            read = open(f, "r")
            lines = read.readlines()
            if lines[0] != '\n' and lines[1] != '\n':  # If it's nonempty
                mdraws.append(np.loadtxt(f))

        # print(mdraws[0].shape); print(len(mdraws))
        mdraws = np.concatenate(mdraws, axis=1)  # Got rid of the transpose

        pred_mean = np.empty(len(mdraws[0]))
        pred_sd = np.empty(len(mdraws[0]))
        pred_5 = np.empty(len(mdraws[0]))
        pred_lower = np.empty(len(mdraws[0]))
        pred_upper = np.empty(len(mdraws[0]))
    
    if readstd:
        sdraw_files = sorted(list(fpath.glob(modelname + ".sdraws*")))
    
        for f in sdraw_files:
            read = open(f, "r")
            lines = read.readlines()
            if lines[0] != '\n' and lines[1] != '\n':  # If it's nonempty
                sdraws.append(np.loadtxt(f))
        # print(sdraws[0]); print(sdraws[0][0])
        # print(len(sdraws)); print(len(sdraws[0])); print(len(sdraws[0][0]))
        sdraws = np.concatenate(sdraws, axis=1)  # Got rid of the transpose

        sigma_mean = np.empty(len(sdraws[0]))
        sigma_sd = np.empty(len(mdraws[0]))
        sigma_5 = np.empty(len(mdraws[0]))
        sigma_lower = np.empty(len(sdraws[0]))
        sigma_upper = np.empty(len(sdraws[0]))
        
        for j in range(len(mdraws[0])):
            pred_mean[j] = np.mean(mdraws[:, j])
            sigma_mean[j] = np.mean(sdraws[:, j])
            pred_sd[j] = np.std(mdraws[:, j], ddof=1)
            sigma_sd[j] = np.std(sdraws[:, j], ddof=1)
            pred_5[j] = np.quantile(mdraws[:, j], 0.50)
            sigma_5[j] = np.quantile(sdraws[:, j], 0.50)
            pred_lower[j] = np.quantile(mdraws[:, j], q_lower)
            sigma_lower[j] = np.quantile(sdraws[:, j], q_lower)
            pred_upper[j] = np.quantile(mdraws[:, j], q_upper)
            sigma_upper[j] = np.quantile(sdraws[:, j], q_upper)