import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


keras = tf.keras

DATASET_PATH = "train/data.json"
SAVED_MODEL_PATH = "cnn.keras"

LEARNING_RATE = 0.001
EPOCHS = 40
BATCH_SIZE = 32


# load data
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    return inputs, targets


def prepare_datasets(path):

    x, y = load_data(path)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train, y_train, test_size=0.2
    )

    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, x_validation, x_test, y_train, y_validation, y_test


def predict(x, y, model):

    x = x[np.newaxis, ...]

    prediction = model.predict(x)

    index = np.argmax(prediction, axis=1)

    print(f"Expected index: {y}, Predicted index: {index}")


def build_model(input_shape, rate, error="sparse_categorical_crossentropy"):

    model = keras.Sequential()

    # conv layer 1
    model.add(
        keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            input_shape=input_shape,
            kernel_regularizer=keras.regularizers.l2(0.001),
        )
    )

    # speed up training and get better results
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # conv layer 2
    model.add(
        keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.001),
        )
    )

    # speed up training and get better results
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # conv layer 3
    model.add(
        keras.layers.Conv2D(
            32,
            (2, 2),
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.001),
        )
    )

    # speed up training and get better results
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # flatten output feed into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # softmax classifier
    model.add(keras.layers.Dense(31, activation="softmax"))

    # optimize

    optimizer = keras.optimizers.Adam(learning_rate=rate)
    model.compile(optimizer=optimizer, loss=error, metrics=["accuracy"])

    # print model overview
    model.summary()

    return model


# convolutional neural network
def main():
    (
        input_train,
        input_validation,
        input_test,
        target_train,
        target_validation,
        target_test,
    ) = prepare_datasets(DATASET_PATH)

    input_shape = (
        input_train.shape[1],
        input_train.shape[2],
        input_train.shape[3],
    )  # (num. segments, num coefficients (13), 1)

    # build the CNN model
    model = build_model(input_shape, LEARNING_RATE)

    # train model
    model.fit(
        input_train,
        target_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(input_validation, target_validation),
    )

    # evaluate
    test_error, test_accuracy = model.evaluate(input_test, target_test)

    print(f"test error: {test_error}, test accuracy: {test_accuracy}")

    # save model

    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()
