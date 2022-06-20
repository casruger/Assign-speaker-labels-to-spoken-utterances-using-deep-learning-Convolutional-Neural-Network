# Assign-speaker-labels-to-spoken-utterances-using-deep-learning-(Convolutional Neural Network)
Predict who is speaking in spoken utterances using a neural network 

# Project description in short
Predict who is speaking in a spoken utterance using an own convolutional neural network. The goal is to reach an as low of an error rate as possible on the unseen test data based on the given test data. 

# Deep Learning Architecture
This project utilizes a Convolutional Neural Network (CNN) as the deep learning model. The
group created the new CNN using Pytorch [2]. The CNN comprises 4 convolutional layers and
2 Max-Pooling layers. In addition, for regularization purposes, after each convolutional layer,
a batch normalization layer has been added. Please find the detailed code in the code file,
and see appendix 1 for the summary of the CNN architecture.

# Training, Hyperparameters, Optimization
First of all, we preprocessed the dataset based on the torchaudio input/output (I/O) tutorial [1].
We can extract features suitable for NN models, such as frequencies and amplitudes of audio
waves. The group also cropped the audios to the same length (2 seconds) and looped the
audio files shorter than 2 seconds to have all audios of the same length. In addition, we applied
the VAD transform from the Pytorch library to remove silent fragments from the audios. Finally,
we transformed them to Mel-Frequency Cepstral Coefficients (MFCC) for training and test
datasets. MFCC was developed based on human perception experiments and enabled more
human-perceptive related representation for the speech audio by adjusting the weight
dimension of frequency [4].In addition, the group also tried data augmentation to reduce
overfitting by adding noise. However, this only led to lower accuracy for both the validation
and test sets predictions. We selected the length of 2 seconds because it gave the best results
as measured by the accuracy of the validation set in further training. In addition, the group
split the training set into training (80% of all training data) and validation data (remaining 20%).
Thus, the optimal parameters can be found in the parameter tuning step using the unseen
validation data. After normalizing the features and trying training with different settings, the
group defined the hyperparameters that gave the best results: the batch size of 16, the
learning rate of 0.001, and the number of epochs of 100, and the group utilized cross-entropy
as the loss function (with Stochastic Gradient Descent as optimizer). In addition, to avoid
overfitting, the group used early-stopping by saving the best-performing models as measured
by the validation loss. After running 100 epochs, it was found that the 92nd epoch had the
best performance. Therefore we continued to use the model that resulted from the 92nd
epoch.

# Performance
The best performing model on the training set had an accuracy score of 1.0 and a validation
set accuracy score of 0.9792. When using this model on the test data and uploading the
predictions to Codalab, the accuracy was 0.9779.


