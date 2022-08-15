# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 23:06:35 2022

@author: peter
"""

#import 
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

#%%
# Making a model for MNIST
# https://www.tensorflow.org/datasets/keras_example
# Used code from above link to get started

#%%

def load_mnist():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        )
    return ds_train, ds_test, ds_info

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def transform_MNIST_train(ds_train, ds_info):
    ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    return ds_train
    
def transform_MNIST_test(ds_test):
    ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    return ds_test

def create_compile_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])
    
    model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    
    return model


def fit_model(model, ds_train, ds_test):
    return model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
    )

def train_model(ds_train, ds_test):
    model = create_model()
    print(model)
    model = fit_model(model, ds_train, ds_test)
    return model
    

def main():
    ds_train, ds_test, ds_info = load_mnist()
    ds_train = transform_MNIST_train(ds_train, ds_info)
    ds_test = transform_MNIST_test(ds_test)
    
    model = train_model(ds_train, ds_test)
    
    return model

if __name__ == "__main__":
    main()
