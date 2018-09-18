import random
import string
import os
from PIL import Image
from claptcha import Claptcha

FONT_PATH = '/tensorflow/captcha/generate_captcha/font'
TEMP_FOLDER = 'temp/'
TRAIN_FOLDER = 'train/'
TEST_FOLDER = 'test/'
VALID_FOLDER = 'validation/'
CAPT_LENGTH = 6

def rdLength():
	return random.randint(6,8)

def rdString():
	chars_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	rndLetters = (random.choice(chars_list) for _ in range(rdLength()))
	return "".join(rndLetters)

def rdFixString():
	chars_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	rndLetters = (random.choice(chars_list) for _ in range(CAPT_LENGTH))
	return "".join(rndLetters)

def randomNoise():
	return float(random.randint(0,10)) / 10

def randomFont():
	list_font = os.listdir(FONT_PATH)
	rndFont = random.choice(list_font) 
	return FONT_PATH + "/" + rndFont

def gen_Fix_Captcha(FOLDER, nb_pic):
	for _ in range(nb_pic):
		''' Fixed length captcha'''
		c = Claptcha(rdFixString, randomFont(), (300,100),
		resample = Image.BILINEAR, noise=randomNoise())
		
		text, _ =c.write(FOLDER + 'temp.png')
		'''	print(text) '''
		os.rename(FOLDER + 'temp.png',FOLDER + text + '.png')

def gen_Captcha(FOLDER, nb_pic):
	for _ in range(nb_pic):
		''' Random length captcha'''
		c = Claptcha(rdString, FONT_PATH+'/Raleway-SemiBold.ttf', (300,100),
		resample = Image.BILINEAR, noise=randomNoise())
		
		text, _ =c.write(FOLDER + 'temp.png')
		'''	print(text) '''
		os.rename(FOLDER + 'temp.png',FOLDER + text + '.png')

gen_Captcha(TRAIN_FOLDER, 100)
