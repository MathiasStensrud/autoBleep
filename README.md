# autoBleep
A neural net enabled video project to read lips and anticipate bad words

## The project
The AutoBleep is a program designed to bleep live video/audio, using video trained on a subjects face. The objective is for the program to detect the first part of a bad word though live video analysis.

## The tech
The project is built off of openCV and keras primarily. An openCV program running a haar classification cascade grabs pictures of a subjects mouth every few frames, then runs them through the classifcation model.
