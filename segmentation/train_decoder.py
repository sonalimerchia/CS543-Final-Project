import tensorflow as tf
import numpy as np
import pickle

from segmentation.loss import softmax_cross_entropy
from segmentation.model import make_segmentation_model

EPOCHS = 400
BATCH_SIZE = 32

def train_segmentation_decoder(data, labels, encoder_names, output_file, hist_file):
    model = make_segmentation_model(data["inputs"][0].shape[-1], 2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-5),
        loss=softmax_cross_entropy, # tf.keras.losses.CategoricalCrossentropy(reduction="sum"),
        metrics=[
            tf.keras.metrics.CategoricalCrossentropy(from_logits=True)
        ]
    )

    model.summary()
    historyMod = model.fit(
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
    
    with open(hist_file, 'wb') as file: 
        pickle.dump(historyMod.history, file)