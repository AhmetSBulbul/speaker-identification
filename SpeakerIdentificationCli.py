import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

def calculate_delta(array):
    rows, cols = array.shape
    deltas = np.zeros((rows, cols))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas

def extract_features(audio, rate):
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined

def record_audio_train():
    Name = input("Please Enter Your Name: ")
    for count in range(5):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 10
        device_index = 2
        audio = pyaudio.PyAudio()
        print("----------------------record device list---------------------")
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
                print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
        print("-------------------------------------------------------------")
        index = int(input("Enter the input device ID: "))
        print("Recording via index ", index)
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, input_device_index=index,
                            frames_per_buffer=CHUNK)
        print("Recording started")
        Recordframes = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        print("Recording stopped")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        OUTPUT_FILENAME = Name + "-sample" + str(count) + ".wav"
        WAVE_OUTPUT_FILENAME = os.path.join("training_set", OUTPUT_FILENAME)
        trainedfilelist = open("training_set_addition.txt", 'a')
        trainedfilelist.write(OUTPUT_FILENAME + "\n")
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()

def record_audio_test():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 10
    device_index = 2
    audio = pyaudio.PyAudio()
    print("----------------------record device list---------------------")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("-------------------------------------------------------------")
    index = int(input("Enter the input device ID: "))
    print("Recording via index ", index)
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=index,
                        frames_per_buffer=CHUNK)
    print("Recording started")
    Recordframes = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print("Recording stopped")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    OUTPUT_FILENAME = "sample.wav"
    WAVE_OUTPUT_FILENAME = os.path.join("testing_set", OUTPUT_FILENAME)
    trainedfilelist = open("testing_set_addition.txt", 'a')
    trainedfilelist.write(OUTPUT_FILENAME + "\n")
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()

def train_model():
    # source = "C:\\Users\\Vaibhav\\Desktop\\SpeakerIdentification\\training_set\\"
    # dest = "C:\\Users\\Vaibhav\\Desktop\\SpeakerIdentification\\trained_models\\"
    # train_file = "C:\\Users\\Vaibhav\\Desktop\\SpeakerIdentification\\training_set_addition.txt"
    current_dir = os.getcwd()
    source = os.path.join(current_dir, "training_set/")
    dest = os.path.join(current_dir, "trained_models/")
    train_file = os.path.join(current_dir, "training_set_addition.txt")

    file_paths = open(train_file, 'r')
    count = 1
    features = np.asarray(())
    labels = []
    for path in file_paths:
        path = path.strip()
        print(path)
        speaker_name = path.split("-")[0]
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
        
        labels.append(speaker_name)

        if count == 5:
            gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=3)
            gmm.fit(features)

            # dumping the trained gaussian model
            picklefile = speaker_name + ".gmm"
            pickle.dump(gmm, open(dest + picklefile, 'wb'))
            print('+ modeling completed for speaker:', picklefile, " with data point =", features.shape)
            
            features = np.asarray(())
            labels = []
            count = 0
        count += 1

def test_model():
    # source = "C:\\Users\\Vaibhav\\Desktop\\SpeakerIdentification\\testing_set\\"
    # modelpath = "C:\\Users\\Vaibhav\\Desktop\\SpeakerIdentification\\trained_models\\"
    # test_file = "C:\\Users\\Vaibhav\\Desktop\\SpeakerIdentification\\testing_set_addition.txt"
    current_dir = os.getcwd()
    source = os.path.join(current_dir, "testing_set/")
    modelpath = os.path.join(current_dir, "trained_models/")
    test_file = os.path.join(current_dir, "testing_set_addition.txt")
    threshold = -25.0
    file_paths = open(test_file, 'r')
    gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    print(gmm_files)
    speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]
    features = np.asarray(())
    labels = []
    # Read the test directory and get the list of test audio files
    for path in file_paths:
        path = path.strip()
        print(path)
        

        sr, audio = read(source + path)
        vector = extract_features(audio, sr)
        labels.append("Ibrahim")  # Add the speaker label

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm = models[i]  # checking with each model one by one
            score = gmm.score(vector)
            print("\n score: ",score)
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
            print("\n spekars: ",speakers)
            print("\n log_likelihood: ",log_likelihood)
            print("\n scores: ",scores)
            

        winner = np.argmax(log_likelihood)
        if log_likelihood[winner] < threshold:
            print("\n Not Recognized! \n")
        else:
            print("\tdetected as - ", speakers[winner])
        time.sleep(1.0)

    accuracy = accuracy_score(labels, [speakers[winner]])
    f_measure = f1_score(labels, [speakers[winner]], average='macro')
    print("Accuracy:", accuracy)
    print("F-Measure:", f_measure)
    return speakers[winner]

if __name__ == '__main__':
    choice = input("Enter Choice Train or Test : ")
    if choice == 'train':
        record_audio_train()
        train_model()
    elif choice == 'test':
        record_audio_test()
        test_model()
    else:
        print("Invalid choice. Please enter either 'train' or 'test'.")
