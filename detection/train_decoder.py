import tensorflow as tf
import numpy as np
import pickle

from detection.model import make_detection_model


EPOCHS = 1000
BATCH_SIZE = 100

def train_segmentation_decoder(data, labels, encoder_names, output_file):
    model = make_detection_model(data["inputs"][0].shape[-1], 9)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-5),
        loss=tf.
        metrics=[
            tf.keras.metrics.CategoricalCrossentropy(from_logits=True)
        ]
    )

    model.summary()
    history = model.fit(
        {"feed1": data["inputs"][0], "feed2": data["inputs"][1], "feed3": data["inputs"][2]},
        {"output": labels},
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1
    )

    predictions = model([data["inputs"][0][:5], data["inputs"][1][:5], data["inputs"][2][:5]], training=False)
    for i, p in enumerate(predictions): 
        np.save(encoder_names[i][1:] + ".npy", p)

    model.save(output_file)
    with open("hist.pkl") as file: 
        pickle.dump(history, file)