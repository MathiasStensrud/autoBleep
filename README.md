# autoBleep
A neural net enabled video project to read lips and anticipate bad words

## The project
The AutoBleep is a program designed to bleep live video/audio, using video trained on a subjects face. The objective is for the program to detect the first part of a bad word though live video analysis.

## The tech
The project is built off of openCV and keras primarily. An openCV program running a haar classification cascade grabs pictures of a subject's mouth every few frames, then runs them through the model.
Haar classifiers are an older computer vision

## The programs
I wrote an assistant program to generate my dataset, which prompts a user with a word for them to pronounce. Once they say the word and press the key to continue, the past 15 frames are recorded, their mouth is found in the image sequence, and a cropped image is saved to the data folder for their respective status as a word to be censored or not.

I also wrote a combination Convolutional Neural Net and Recurrent Neural Net, that took in the image sequences and preformed a small amount of predictions upon them before passing them into a combined LSTM predictor.
