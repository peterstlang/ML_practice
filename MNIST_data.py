# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 23:06:35 2022

@author: peter
"""

#import 
import tensorflow as tf
import tensorflow_datasets as tfds

#%%
# Making a model for MNIST
# https://www.tensorflow.org/datasets/keras_example
# Used code from above link to get started

#%%

def load_dataset(dataset='mnist'):
    
    if dataset in tfds.list_builders():
        builder = tfds.builder(dataset)
        builder.download_and_prepare()
        ds = builder.as_dataset(split='train', shuffle_files=True)
        return ds
    else:
        print("{} is not a valid dataset!".format(dataset))
        
def iterate_examples(ds, num_examples=3):
    ds = ds.take(num_examples)
    
    for example in ds:
        print(list(example.keys()))
        image = example["image"]
        label = example["label"]
        print(image.shape, label)
    
def benchmark(ds, batch_size=32):
    ds = ds.batch(batch_size).prefetch(1)

    tfds.benchmark(ds, batch_size=batch_size)
    tfds.benchmark(ds, batch_size=batch_size)

if __name__ == "__main__":

    #ds = load_dataset()
    #iterate_examples(num_examples=10)
    #benchmark(batch_size=64)
    ds, info = tfds.load('mnist', split='train', with_info=True)

    df = tfds.as_dataframe(ds.take(1), info)
    

    
    
    