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

def main():
    ds, info = tfds.load('mnist', split='train', with_info=True)
    fig = tfds.show_examples(ds, info)
    

if __name__ == "__main__":
    main()
