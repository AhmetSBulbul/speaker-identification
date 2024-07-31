import os
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score
import pickle

from speaker_identification.__utils import extract_features


def train_model():
    # root_dir = os.path.dirname(os.getcwd())
    root_dir = os.getcwd()
    source = os.path.join(root_dir, "data_set/")
    dest = os.path.join(root_dir, "trained_models/")
    speaker_list = os.listdir(source)
    # Training the GMMs
    for speaker in speaker_list:
        path = os.path.join(source, speaker)
        train_path = os.path.join(path, "train_set/")
        files = [os.path.join(train_path, fname) for fname in os.listdir(train_path) if fname.endswith('.wav')]
        features = np.asarray(())
        for file in files:
            sr, audio = read(file)
            vector = extract_features(audio, sr)
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
        gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=3)
        gmm.fit(features)

        # dumping the trained gaussian model
        picklefile = speaker + ".gmm"
        pickle.dump(gmm, open(dest + picklefile, 'wb'))
        print('+ modeling completed for speaker:', picklefile, " with data point =", features.shape)

    # Testing the data
    labels = []
    winners = []
    for speaker in speaker_list:
        path = os.path.join(source, speaker)
        test_path = os.path.join(path, "test_set/")
        files = [os.path.join(test_path, fname) for fname in os.listdir(test_path) if fname.endswith('.wav')]
        gmm_files = [os.path.join(dest, fname) for fname in os.listdir(dest) if fname.endswith('.gmm')]
        models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
        speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]
        for file in files:
            sr, audio = read(file)
            vector = extract_features(audio, sr)
            log_likelihood = np.zeros(len(models))
            labels.append(speaker)
            for i in range(len(models)):
                gmm = models[i]
                score = gmm.score(vector)
                scores = np.array(score)
                log_likelihood[i] = scores.sum()
            winner = np.argmax(log_likelihood)
            #TODO: Add threshold
            winners.append(speakers[winner])
            print("\nLabel: ", speaker, "Winner: ", speakers[winner], "Log Likelihood: ", log_likelihood)
    print("\nLabels: ", labels)
    print("\nWinners: ", winners)
    accuracy = accuracy_score(labels, winners)
    f_measure = f1_score(labels, winners, average='macro')
    print("\nAccuracy: ", accuracy)
    print("\nF Measure: ", f_measure)
    return accuracy, f_measure



    