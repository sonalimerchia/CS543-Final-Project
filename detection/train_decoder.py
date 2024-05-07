import tensorflow as tf
import numpy as np
import pickle

from detection.model import make_detection_model
from detection.loss import box_loss, confidence_loss, delta_conf_loss, delta_boxes_loss

EPOCHS = 400
BATCH_SIZE = 32

def train_detection_decoder(data, labels, encoder_names, output_file, hist_file):
    model = make_detection_model(data["inputs"][0].shape[-1], 9)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-5),
        losses=[box_loss, confidence_loss, delta_conf_loss, delta_boxes_loss],
    )

    model.summary()
    
    historyMod = model.fit(
        {"feed1": data["inputs"][0], "feed2": data["inputs"][1]},
        {"output": labels},
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1
    )

    predictions = model([data["inputs"][0][:5], data["inputs"][1][:5], data["inputs"][2][:5]], training=False)
    for i, p in enumerate(predictions): 
        np.save(encoder_names[i][1:] + ".npy", p)
    model.save(output_file)

    with open(hist_file) as file: 
        pickle.dump(historyMod.history, file)