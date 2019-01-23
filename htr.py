#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 13:19:20 2018

@author: adarsh
"""

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import random

#files=os.listdir('images/')
#random.shuffle(files)
#img_pix=np.zeros((30000,2500))
#labels=[]
#for i in range(30000):
#    img=Image.open('images/'+files[i], mode='r')
#    npimg=np.array(img)
#    img_pix[i]=npimg.flatten()
#    labels.append("#"+files[i].split('_')[0]+"@")
#    
#labels=np.array(labels, dtype=str)
#labels=labels.reshape((30000,1))
#
#
#plt.imshow(img_pix[5].reshape((25,100)), cmap='gray')


from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Bidirectional, Concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


batch_size = 128  # Batch size for training.
epochs = 20  # Number of epochs to train for.
latent_dim = 128  # Latent dimensionality of the encoding space.
num_samples = 30000  # Number of samples to train on.
# Path to the data txt file on disk.
#data_path = 'fra-eng/fra.txt'

#load images and labels
labels=pd.read_pickle('labels.pkl')
images=pd.read_pickle('pixels.pkl')
labels=list(labels['word'])
images=np.array(images, dtype=int)
images=images.reshape(30000,25,100,1)
images=images/255
# Vectorize the data.
#input_texts = []
target_texts =labels

target_characters = list('abcdefghijklmnopqrstuvwxyz@#')
#with open(data_path, 'r', encoding='utf-8') as f:
#    lines = f.read().split('\n')

#input_text=array of all input lines(10000,max no of characters in a line)
#output_test=array of all output lines

                         
#for line in lines[: min(num_samples, len(lines) - 1)]:
#    input_text, target_text = line.split('\t')
#    # We use "tab" as the "start sequence" character
#    # for the targets, and "\n" as "end sequence" character.
#    target_text = '\t' + target_text + '\n'
#    input_texts.append(input_text)
#    target_texts.append(target_text)
#    for char in input_text:
#        if char not in input_characters:
#            input_characters.add(char)
#    for char in target_text:
#        if char not in target_characters:
#            target_characters.add(char)

#input_characters = sorted(list(input_characters))
#target_characters = sorted(list(target_characters))
#num_encoder_tokens = len(input_characters)

num_decoder_tokens = len(target_characters)
#max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

#print('Number of samples:', len(input_texts))
#print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
#print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

#input_token_index = dict(
#    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = images
decoder_input_data = np.zeros(
    (30000, max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (30000, max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, target_text in enumerate( target_texts):
    
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# Define an input sequence and process it.
cnn_inputs = Input(shape=(25,100,1))
conv1=Conv2D(32, kernel_size=(5, 5), activation='relu')(cnn_inputs)
conv2=Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
maxpool=MaxPooling2D(pool_size=(2, 2))(conv2)
dropout=Dropout(0.5)(maxpool)
flatten=Flatten()(dropout)
imgvec=Dense(128, activation='relu')(flatten)
#imgcon=Concatenate(axis=-1)([imgvec,imgvec])
#imgvec=Reshape((2,128))(imgcon)
#
##encoder_inputs = Input(shape=(None, num_encoder_tokens))
#encoder = LSTM(latent_dim, return_state=True)
#encoder_outputs, state_h, state_c = encoder(imgvec)
## We discard `encoder_outputs` and only keep the states.
#encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=[imgvec,imgvec])
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([cnn_inputs, decoder_inputs], decoder_outputs)

# Run training

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('htr__.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(cnn_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
#reverse_input_char_index = dict(
#    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['#']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '@' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = images[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Decoded sentence:', decoded_sentence)


from PIL import Image
img=Image.open('test_sans.png')
img=np.reshape(img,(1,25,100,1))/255
decoded_sentence = decode_sequence(img)
print('-')
print('Decoded sentence:', decoded_sentence)