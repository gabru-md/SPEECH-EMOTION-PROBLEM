import pandas as pd
from fastai.vision import *

import argparse
import librosa
import librosa.display
import numpy as np
import os

learner_path = "meld/"
test_img_folder = "meld/test_img"
hop_length = 512
window_size = 1024
sr = 8000 # human voice range does not exceed 8khz; will be used for resampling

# computes a stft spectrogram for an audio file
def stft_spec(folder, wav_file):
  y, o_sr = librosa.load(os.path.join(folder, wav_file))
  y = librosa.resample(y, o_sr, sr)
  window = np.hanning(window_size)
  out  = librosa.core.spectrum.stft(y, n_fft = window_size, hop_length = hop_length, 
       window=window)
  out = 2 * np.abs(out) / np.sum(window)
  return librosa.display.specshow(librosa.amplitude_to_db(out,ref=np.max))

#converts a wav file in a folder to a png image ans saves it to a target
def cvt_wav_2_img(folder, wav_file, target):
  # loads the wav_file and saves it as the required spectrogram
  import pylab
  pylab.axis('off') # no axis
  pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
  stft_spec(folder, wav_file)
  pylab.savefig(os.path.join(target, os.path.splitext(wav_file)[0] + ".png"), bbox_inches=None, pad_inches=0)
  pylab.close()


def cvt_test_data(test_folder):
	for wav_file in os.listdir(test_folder):
		cvt_wav_2_img(test_folder, wav_file, test_img_folder)
	print("[!] done converting the testing data")

def run_save_pred(test_img_folder):
	test = ImageList.from_folder(test_img_folder)
	print("[!] %d samples found."%(len(test)))
	learn = load_learner(learner_path, test=test)
	preds, _ = learn.get_preds(ds_type=DatasetType.Test)
	labelled_preds = [learn.data.classes[np.argmax(pred)] for pred in preds]
	print(labelled_preds)
	fnames = [f.name[:-4] for f in learn.data.test_ds.items]

	df = pd.DataFrame({'File name': fnames, 'prediction': labelled_preds}, columns=['File name', 'prediction'])
	df.to_csv("submission.csv", index=False)
	print("[!] submission.csv created")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Processes the testing data file for Winter Internship Task - MIDAS IIITD\n Usage: python3 test_data.py -i <location of test folder>')
	parser.add_argument('-i', help='the input folder')

	args = parser.parse_args()
	test_folder = args.i
	cvt_test_data(test_folder)
	run_save_pred(test_img_folder)