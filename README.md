# autoBleep
A neural net enabled video project to read lips, anticipate bad words and put a stop to that rudeness.

## The project
The AutoBleep is a program designed to bleep live video/audio, using video trained on a subjects face. The objective is for the program to detect the first part of a bad word though live video analysis, then play a loud sound before the word is finished, effectivly censoring the curse word in real time.

## The tech
The project is built off of openCV and keras primarily. An openCV program running a haar classification cascade grabs pictures of a subject's mouth every few frames, then runs them through the model. The model itself uses the VGG16 neural net model, but runs every image in a sequence through it before sending all of these weights into a bidirectional LSTM predictor.

## The programs
I wrote an assistant program to generate my dataset, which prompts a user with a word for them to pronounce. Once they say the word and press the key to continue, the past 15 frames are recorded, their mouth is found in the image sequence, and a cropped image is saved to the data folder for their respective status as a word to be censored or not.

## The Terminology
### Haar Cascades
Haar Cascades are sets of Haar-like features, which are very weak classifiers individually, but which can compute at incredible speed. These are sets of rectangles of various intensities that are compared together to predict features.
Haar cascades were used in the first real time face detector, opeing up a new area of possibilites in computer vison and image classification. The main benefit is their very quick classification speed compares to almost any other image classifier. They have been integrated with OpenCV to a great deal, aiding in thier use.

### Recurrent Neural Networks and LSTM Layers
God, this is the _fun_ stuff. Neural nets are seen as a bit of a 'black box' by a lot of people, and that also adds to their allure. While theyr are specially equipped to deal with certain problems, they are most definetly not an overall solution.
One problem that they excel at is image classification.

### Neural Nets and Transfer Learning
Training an accurate neural net generally takes a great deal of time, and also requires alarge amount of training data.
