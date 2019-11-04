## SPEECH-EMOTION-PROBLEM

Speech-Emotion-Problem aims at solving the classification of voice samples from the MELD dataset. The data provided by MIDAS-IIITD is a subset of the dataset in MELD. This dataset contains samples of audio in .wav format and is divided into five categories or classes

- happy
- neutral
- sad
- disgust
- fear

Since the dataset is audio .wav files hence we need to define a way to perform classification over the audio files.

Related iPython notebook : [**Midas_Task.ipynb**](https://github.com/gabru-md/SPEECH-EMOTION-PROBLEM/blob/master/Midas_task.ipynb)

### Using Spectrograms
A spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies with time. When applied to an audio signal, spectrograms are sometimes called sonographs, voiceprints, or voicegrams.

### Use of Spectrogram

It is difficult to perform classification on audio samples and the methodologies are not well known, but we do know how to perform classification on images since image classification is a hot topic and the methods are pretty well-known and readily available. Since an audio file can be represented into a corresponding image of its spectrogram therefore we can simply convert the entire dataset into images and then use the newly formed dataset to train our model and perform image recognition on the converted audio files.

The saved models are in the google drive and the link to which will be added once it is uploaded.

### Testing New Data

To use the `test_data.py` simply activate the conda environment using the `create_env` bash file and type

- `python3 test_data.py -i <folder_containing_test_files>`

and it will generate a solution file named as `submissions.csv` in the main folder.

### Kaggle Solution - Planet Problem

Since the solution made on kaggle was a late submission as the contest had already ended long back, therefore it may not be possible to find my solution
on kaggle therefore this repository contains the iPython notebook for the Planet Problem on Kaggle as well to demonstrate the work 
that was submitted.

Related iPython notebook : [**Planet_Midas.ipynb**](https://github.com/gabru-md/SPEECH-EMOTION-PROBLEM/blob/master/Planet_Midas.ipynb)


**Note** : *No saved model was uploaded since github didn't allow uploading files as big as the trained model. No data has been uploaded
in this repository and it contains only the required iPython Notebooks for the two solutions.*


**Author - Manish Devgan**
