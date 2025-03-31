import math
import os
import librosa, librosa.display
import json


DATASET_PATH = "dataset/speech_commands_v0.01"
JSON_PATH = "data.json"


SR = 22050
DURATION = 1  # in seconds
SAMPLES_PER_TRACK = SR * DURATION


def save_mfcc(data_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512):
    data = {"mapping": [], "mfcc": [], "labels": [], "path": []}

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_path)):

        if dirpath is not data_path:
            # save the speech command
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)

            print(f"Processing {label}")

            for f in filenames:

                path = os.path.join(dirpath, f)
                signal, sr = librosa.load(path, sr=SR)

                # extract the mfcc and store data

                if len(signal) >= SAMPLES_PER_TRACK:

                    # enforce 1 sec. long signal

                    signal = signal[:SAMPLES_PER_TRACK]

                    mfcc = librosa.feature.mfcc(
                        y=signal,
                        sr=sr,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mfcc=n_mfcc,
                    )

                    mfcc = mfcc.T
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i - 1)
                    data["path"].append(path)
                    print(f"{path}: {i-1}")
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
