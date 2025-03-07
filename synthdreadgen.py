import random
import string
import os
import shutil
import uuid
from captcha.image import ImageCaptcha

import itertools

def random_color():
    return (random.randint(128,255),
            random.randint(128, 255), 
            random.randint(128, 255),
           255)
PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'dread')
width = 129
height = 40
image = ImageCaptcha(width=width, height=height, fonts=['aptos.ttf'], font_sizes=[20,22,23])
image.character_rotate=(0,0)
image.word_space_probability = 0.01
image.character_offset_dx=(40,60)
image.character_offset_dy=(20,100)
img_dir = os.path.join(DATA_PATH, 'train')
if os.path.exists(img_dir):
    shutil.rmtree(img_dir)
if not os.path.exists(img_dir):
    os.makedirs(img_dir) 
for i in itertools.permutations([str(c) for c in range(10)], 6):
    # This is the label with spaces for the image
    display_text = " " + " ".join(i)
    # This is the label without spaces for the filename
    label = ''.join(i)
    fn = os.path.join(img_dir, f'{label}.png')
    color = random_color()
    image.write(display_text, fn, bg_color=(62, 70, 88), fg_color=color)
