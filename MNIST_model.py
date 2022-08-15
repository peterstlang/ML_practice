# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 23:19:55 2022

@author: peter
"""
from MNIST_data import splits
import tensorflow as tf
import numpy as np


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
    model = create_compile_model()
    model = fit_model(model, ds_train, ds_test)
    return model

if __name__ == "__main__":
    splits()