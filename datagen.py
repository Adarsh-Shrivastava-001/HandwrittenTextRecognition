#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 22:08:35 2018

@author: adarsh
"""

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

data=pd.read_csv('google-10000-english.txt', header=None)

for i in range(5000,10000):
    img=Image.new('RGBA', (200,50), 'white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('GreatVibes-Regular.otf', size=50)
    word=data[0][i]
    draw.text((5,0), word, font=font, fill='black')
    name="images1/"+word+"_gv.png"
    plt.imshow(img)
    img.save(name)
    

for i in range(5000,10000):   
    img=Image.new('RGBA', (200,50), 'white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('Pacifico.ttf', size=30)
    word=data[0][i]
    draw.text((5,0), word, font=font, fill='black')
    name="images1/"+word+"_pac.png"
    plt.imshow(img)
    img.save(name)

for i in range(5000,10000):
    img=Image.new('RGBA', (200,50), 'white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('Sans.otf', size=30)
    word=data[0][i]
    draw.text((5,5), word, font=font, fill='black')
    name="images1/"+word+"_sans.png"
    plt.imshow(img)
    img.save(name)

