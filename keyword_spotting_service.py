import librosa
import numpy as np
import tensorflow as tf

keras = tf.keras

SAMPLES_TO_CONSIDER = 22050


class Keyword_Spotting_Service:
    model = None
    mappings = [
        "right",
        "eight",
        "cat",
        "tree",
        "bed",
        "happy",
        "go",
        "dog",
        "no",
        "wow",
        "nine",
        "left",
        "stop",
        "three",
        "_background_noise_",
        "sheila",
        "one",
        "bird",
        "zero",
        "seven",
        "up",
        "marvin",
        "two",
        "house",
        "down",
        "six",
        "yes",
        "on",
        "five",
        "off",
        "four",
    ]
    instance = None

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):

        signal, sr = librosa.load(file_path)

        if len(signal) > SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]

        MFCCs = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mfcc=n_mfcc,
        )
        return MFCCs.T

    def predict(self, file_path):

        MFCCs = self.preprocess(file_path)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        predictions = self.model.predict(MFCCs)

        keyword = self.mappings[np.argmax(predictions)]

        return keyword


def start_service():
    service = Keyword_Spotting_Service()
    service.model = keras.models.load_model("cnn.h5")
    return service


if __name__ == "__main__":
    service = start_service()
    keyword_1 = service.predict("test/bed1.wav")
    print(f"Predicted keyword: {keyword_1}")
