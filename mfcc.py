import os
import pandas as pd
import librosa 
import numpy as np

# Set the path to the directory containing the audio files
audio_path = "testAudio/"

# Define a function to extract MFCCs from an audio file
def extract_mfcc(audio_file):
    y, sr = librosa.load(audio_file, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    mfccs_all = np.vstack((mfccs, mfccs_delta, mfccs_delta2))
    return mfccs_all.T


# Initialize empty lists to store the MFCCs, labels, and dates/times
mfccs_data = []
labels = []
dates = []
times = []

# Loop through all the audio files in the directory and extract their MFCCs
for filename in os.listdir(audio_path):
    if filename.endswith(".mp3"):
        file_path = os.path.join(audio_path, filename)
        label = filename.split("_")[5] + "_" + filename.split(".")[0].split("_")[6]  # extract the label from the filename
        date_time = filename.split("_")[0] + "_" + filename.split("_")[1] + "_" + filename.split("_")[2] + "_" + \
                    filename.split("_")[3] + "_" + filename.split("_")[4]  # extract the date and time from the filename
        mfccs = extract_mfcc(file_path)
        mfccs_avg = np.mean(mfccs, axis=0)  # compute the average MFCCs across time
        mfccs_data.append(mfccs_avg)
        labels.append(label)
        dates.append(date_time.split("_")[0])
        times.append(date_time.split("_")[1] + "_" + date_time.split("_")[2] + "_" + date_time.split("_")[3] + "_" +
                     date_time.split("_")[4])

# Convert the MFCCs, labels, dates, and times to data frames using pandas
mfccs_df = pd.DataFrame(mfccs_data, columns=["mfcc_" + str(i) for i in range(39)])
labels_df = pd.DataFrame(labels, columns=["label"])
dates_df = pd.DataFrame(dates, columns=["date"])
times_df = pd.DataFrame(times, columns=["time"])

# Concatenate the MFCCs, labels, dates, and times into a single data frame
data = pd.concat([mfccs_df, dates_df, times_df, labels_df], axis=1)

# Save the data frame to a CSV file
data.to_csv("mfcc_data.csv", index=False)
