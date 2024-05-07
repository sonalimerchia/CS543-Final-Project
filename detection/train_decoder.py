import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pickle

from detection.model import make_detection_model
from detection.loss import box_loss, confidence_loss, delta_conf_loss, roi_box_loss

EPOCHS = 100
BATCH_SIZE = 32

def train_detection_decoder(data, labels, first_feed, encoder_names, output_file, hist_file):
    model = make_detection_model(data["inputs"][first_feed].shape[-1], 2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-5),
        loss=[box_loss, confidence_loss, delta_conf_loss, roi_box_loss],
    )

    model.summary()
    labels = tf.convert_to_tensor(labels)

    historyMod = model.fit(
        {"feed1": data["inputs"][first_feed], "feed2": data["inputs"][2]},
        labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1
    )

    predictions = model([data["inputs"][first_feed][:5], data["inputs"][2][:5]], training=False)
    for i, p in enumerate(predictions): 
        np.save(encoder_names[i][1:] + ".npy", p)
    model.save(output_file)

    with open(hist_file) as file: 
        pickle.dump(historyMod.history, file)