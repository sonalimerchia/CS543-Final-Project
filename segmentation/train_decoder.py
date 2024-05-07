import tensorflow as tf
import numpy as np
import pickle

from segmentation.loss import softmax_cross_entropy
from segmentation.model import make_segmentation_model

EPOCHS = 100
BATCH_SIZE = 32

def train_segmentation_decoder(data, labels, first_feed, encoder_names, output_file, hist_file):
    tf.keras.utils.get_custom_objects()['custom_loss'] = softmax_cross_entropy
    labels = tf.convert_to_tensor(labels)

    model = make_segmentation_model(data["inputs"][first_feed].shape[-1], 2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-5),
        loss=softmax_cross_entropy, # tf.keras.losses.CategoricalCrossentropy(reduction="sum"),
        metrics=[
            tf.keras.metrics.CategoricalCrossentropy(from_logits=True)
        ]
    )

    model.summary()

    historyMod = model.fit(
        {"feed1": data["inputs"][first_feed], "feed2": data["inputs"][2], "feed3": data["inputs"][3]},
        {"output": labels},
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1
    )

    predictions = model([data["inputs"][first_feed][:5], data["inputs"][2][:5], data["inputs"][3][:5]], training=False)
    for i, p in enumerate(predictions): 
        np.save(encoder_names[i][1:] + ".npy", p)
    model.save(output_file)
    
    with open(hist_file, 'wb') as file: 
        pickle.dump(historyMod.history, file)