#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn 
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torchvision
import random
import json
from torch.utils.data import DataLoader, TensorDataset


random.seed(373)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device: {}".format(device))
print(torch.__version__)
print(torchaudio.__version__)


# In[16]:


#-------------------------------------------------------------------------------
# Preparation of data and helper functions.
# Source: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
#-------------------------------------------------------------------------------
import io
import os
import math
import tarfile
import multiprocessing

import scipy
import librosa
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import requests
import matplotlib
import matplotlib.pyplot as plt
import time
from IPython.display import Audio, display

[width, height] = matplotlib.rcParams['figure.figsize']
if width < 10:
  matplotlib.rcParams['figure.figsize'] = [width * 2.5, height]

def _get_sample(path, resample=None):
  return path

def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)

def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

def inspect_file(path):
  print("-" * 10)
  print("Source:", path)
  print("-" * 10)
  print(f" - File size: {os.path.getsize(path)} bytes")
  print(f" - {torchaudio.info(path)}")

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

def plot_mel_fbank(fbank, title=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Filter bank')
  axs.imshow(fbank, aspect='auto')
  axs.set_ylabel('frequency bin')
  axs.set_xlabel('mel bin')
  plt.show(block=False)

def get_spectrogram(
    n_fft = 400,
    win_len = None,
    hop_len = None,
    power = 2.0,
):
  waveform, _ = get_speech_sample()
  spectrogram = T.Spectrogram(
      n_fft=n_fft,
      win_length=win_len,
      hop_length=hop_len,
      center=True,
      pad_mode="reflect",
      power=power,
  )
  return spectrogram(waveform)

def plot_pitch(waveform, sample_rate, pitch):
  figure, axis = plt.subplots(1, 1)
  axis.set_title("Pitch Feature")
  axis.grid(True)

  end_time = waveform.shape[1] / sample_rate
  time_axis = torch.linspace(0, end_time,  waveform.shape[1])
  axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

  axis2 = axis.twinx()
  time_axis = torch.linspace(0, end_time, pitch.shape[1])
  ln2 = axis2.plot(
      time_axis, pitch[0], linewidth=2, label='Pitch', color='green')

  axis2.legend(loc=0)
  plt.show(block=False)

def plot_kaldi_pitch(waveform, sample_rate, pitch, nfcc):
  figure, axis = plt.subplots(1, 1)
  axis.set_title("Kaldi Pitch Feature")
  axis.grid(True)

  end_time = waveform.shape[1] / sample_rate
  time_axis = torch.linspace(0, end_time,  waveform.shape[1])
  axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

  time_axis = torch.linspace(0, end_time, pitch.shape[1])
  ln1 = axis.plot(time_axis, pitch[0], linewidth=2, label='Pitch', color='green')
  axis.set_ylim((-1.3, 1.3))

  axis2 = axis.twinx()
  time_axis = torch.linspace(0, end_time, nfcc.shape[1])
  ln2 = axis2.plot(
      time_axis, nfcc[0], linewidth=2, label='NFCC', color='blue', linestyle='--')

  lns = ln1 + ln2
  labels = [l.get_label() for l in lns]
  axis.legend(lns, labels, loc=0)
  plt.show(block=False)

DEFAULT_OFFSET = 201
SWEEP_MAX_SAMPLE_RATE = 48000
DEFAULT_LOWPASS_FILTER_WIDTH = 6
DEFAULT_ROLLOFF = 0.99
DEFAULT_RESAMPLING_METHOD = 'sinc_interpolation'

def _get_log_freq(sample_rate, max_sweep_rate, offset):
  """Get freqs evenly spaced out in log-scale, between [0, max_sweep_rate // 2]

  offset is used to avoid negative infinity `log(offset + x)`.

  """
  half = sample_rate // 2
  start, stop = math.log(offset), math.log(offset + max_sweep_rate // 2)
  return torch.exp(torch.linspace(start, stop, sample_rate, dtype=torch.double)) - offset

def _get_inverse_log_freq(freq, sample_rate, offset):
  """Find the time where the given frequency is given by _get_log_freq"""
  half = sample_rate // 2
  return sample_rate * (math.log(1 + freq / offset) / math.log(1 + half / offset))

def _get_freq_ticks(sample_rate, offset, f_max):
  # Given the original sample rate used for generating the sweep,
  # find the x-axis value where the log-scale major frequency values fall in
  time, freq = [], []
  for exp in range(2, 5):
    for v in range(1, 10):
      f = v * 10 ** exp
      if f < sample_rate // 2:
        t = _get_inverse_log_freq(f, sample_rate, offset) / sample_rate
        time.append(t)
        freq.append(f)
  t_max = _get_inverse_log_freq(f_max, sample_rate, offset) / sample_rate
  time.append(t_max)
  freq.append(f_max)
  return time, freq

def plot_sweep(waveform, sample_rate, title, max_sweep_rate=SWEEP_MAX_SAMPLE_RATE, offset=DEFAULT_OFFSET):
  x_ticks = [100, 500, 1000, 5000, 10000, 20000, max_sweep_rate // 2]
  y_ticks = [1000, 5000, 10000, 20000, sample_rate//2]

  time, freq = _get_freq_ticks(max_sweep_rate, offset, sample_rate // 2)
  freq_x = [f if f in x_ticks and f <= max_sweep_rate // 2 else None for f in freq]
  freq_y = [f for f in freq if f >= 1000 and f in y_ticks and f <= sample_rate // 2]

  figure, axis = plt.subplots(1, 1)
  axis.specgram(waveform[0].numpy(), Fs=sample_rate)
  plt.xticks(time, freq_x)
  plt.yticks(freq_y, freq_y)
  axis.set_xlabel('Original Signal Frequency (Hz, log scale)')
  axis.set_ylabel('Waveform Frequency (Hz)')
  axis.xaxis.grid(True, alpha=0.67)
  axis.yaxis.grid(True, alpha=0.67)
  figure.suptitle(f'{title} (sample rate: {sample_rate} Hz)')
  plt.show(block=True)

def get_sine_sweep(sample_rate, offset=DEFAULT_OFFSET):
    max_sweep_rate = sample_rate
    freq = _get_log_freq(sample_rate, max_sweep_rate, offset)
    delta = 2 * math.pi * freq / sample_rate
    cummulative = torch.cumsum(delta, dim=0)
    signal = torch.sin(cummulative).unsqueeze(dim=0)
    return signal

def benchmark_resample(
    method,
    waveform,
    sample_rate,
    resample_rate,
    lowpass_filter_width=DEFAULT_LOWPASS_FILTER_WIDTH,
    rolloff=DEFAULT_ROLLOFF,
    resampling_method=DEFAULT_RESAMPLING_METHOD,
    beta=None,
    librosa_type=None,
    iters=5
):
  if method == "functional":
    begin = time.time()
    for _ in range(iters):
      F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=lowpass_filter_width,
                 rolloff=rolloff, resampling_method=resampling_method)
    elapsed = time.time() - begin
    return elapsed / iters
  elif method == "transforms":
    resampler = T.Resample(sample_rate, resample_rate, lowpass_filter_width=lowpass_filter_width,
                           rolloff=rolloff, resampling_method=resampling_method, dtype=waveform.dtype)
    begin = time.time()
    for _ in range(iters):
      resampler(waveform)
    elapsed = time.time() - begin
    return elapsed / iters
  elif method == "librosa":
    waveform_np = waveform.squeeze().numpy()
    begin = time.time()
    for _ in range(iters):
      librosa.resample(waveform_np, sample_rate, resample_rate, res_type=librosa_type)
    elapsed = time.time() - begin
    return elapsed / iters


# In[17]:


# Import Train and Test dictionary

f = open('train.json')
train = json.load(f)

g = open('test.json')
test = json.load(g)


# In[18]:


# Create Test and Train DataFrames with path of Audio files

train_dir = 'data/train/'

df_train = pd.DataFrame.from_dict(train, orient='index', columns = ['Label'])
df_train.reset_index(inplace=True)
df_train = df_train.rename(columns={'index': 'Path'}) 
df_train['Path'] = train_dir + df_train['Path']

test_dir = 'data/test/'

df_test = pd.DataFrame.from_dict(test, orient='index', columns = ['Label'])
df_test.reset_index(inplace=True)
df_test = df_test.rename(columns={'index': 'Path'}) 
df_test['Path'] = test_dir + df_test['Path']

df_train


# In[34]:


# Create targets' list for training and compute number of classes

train_labels = df_train['Label'].tolist()
train_labels = [int(i) for i in train_labels]

print('Number of Classes:', len(set(train_labels)))


# In[38]:


#-------------------------------------------------------------------------------
# FEATURES EXTRACTION FOR TRAINING
# Pad or crop the audios to same length (2 seconds)
# Transform to MFCC which will constitute the features of the model
#-------------------------------------------------------------------------------

train_features = []
length = 32000

for path in df_train['Path'].tolist():
    waveform, sample_rate = torchaudio.load(path)
    vad = torchaudio.transforms.Vad(sample_rate)
    waveform = vad(waveform)
    if waveform.shape[1] < length:
        waveform = torch.cat(((waveform,)*((length//waveform.shape[1])+1)),1)
        crop = torchvision.transforms.RandomCrop((1,length))
        waveform = crop(waveform)
    else:
        crop = torchvision.transforms.RandomCrop((1,length))
        waveform = crop(waveform)

    n_fft = 2048
    win_length = None
    hop_length = 512
    n_mels = 256
    n_mfcc = 256

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
          'n_fft': n_fft,
          'n_mels': n_mels,
          'hop_length': hop_length,
          'mel_scale': 'htk',
        }
    )

    mfcc = mfcc_transform(waveform)
    train_features.append(mfcc)


# In[39]:


#-------------------------------------------------------------------------------
# FEATURES EXTRACTION FOR TESTING
# Pad or crop the audios to same length (1 second)
# Transform to MFCC which will constitute the features of the model
#-------------------------------------------------------------------------------

test_features = []
length = 32000

for path in df_test['Path'].tolist():
    waveform, sample_rate = torchaudio.load(path)
    vad = torchaudio.transforms.Vad(sample_rate)
    waveform = vad(waveform)
    if waveform.shape[1] < length:
        waveform = torch.cat(((waveform,)*((length//waveform.shape[1])+1)),1)
        crop = torchvision.transforms.RandomCrop((1,length))
        waveform = crop(waveform)
    else:
        crop = torchvision.transforms.RandomCrop((1,length))
        waveform = crop(waveform)

    n_fft = 2048
    win_length = None
    hop_length = 512
    n_mels = 256
    n_mfcc = 256

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
          'n_fft': n_fft,
          'n_mels': n_mels,
          'hop_length': hop_length,
          'mel_scale': 'htk',
        }
    )

    mfcc = mfcc_transform(waveform)
    test_features.append(mfcc)


# In[40]:


# Split dataset in train and validation set
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=373)

print('# Training examples: {}'.format(len(X_train)))
print('# Validation examples: {}'.format(len(X_val)))
print('# Training targets: {}'.format(len(y_train)))
print('# Validation targets: {}'.format(len(y_val)))


# In[ ]:


# Normalize features
X_train_tensor_mean = torch.mean(X_train_tensor)
X_train_tensor_std = torch.std(X_train_tensor, unbiased=False)

normalize = torchvision.transforms.Normalize(X_train_tensor_mean, X_train_tensor_std)

X_train_tensor = normalize(X_train_tensor)
X_val_tensor = normalize(X_val_tensor)
X_test_tensor = normalize(X_test_tensor)


# In[ ]:


# Define relevant variables and create datasets and dataloaders
batch_size = 16
num_classes = 184 # number of classes + 1
learning_rate = 0.001
num_epochs = 100

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# In[ ]:


# Part of code adapted from: https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/
# Creating a CNN class
class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(46848, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.batch_norm1(out)
        out = self.conv_layer2(out)
        out = self.batch_norm2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.batch_norm3(out)
        out = self.conv_layer4(out)
        out = self.batch_norm4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


# In[ ]:


model = ConvNeuralNet(num_classes)

# Set Loss function
criterion = nn.CrossEntropyLoss()

# Set optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

total_step = len(train_loader)

# Move model to device
model = model.to(device)


# In[ ]:


# We use the pre-defined number of epochs to determine how many iterations to train the network on
# Part of code adapted from: https://www.geeksforgeeks.org/training-neural-networks-with-validation-using-pytorch/
min_valid_loss = np.inf

for epoch in range(num_epochs):
	#Load in the data in batches using the train_loader object
    train_loss = 0.0    
    for files, labels in train_loader:
        # Move tensors to the configured device
        files = files.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(files)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate Loss
        train_loss += loss.item()
    
    valid_loss = 0.0
    correct = 0
    total = 0
    model.eval()     
    for files, labels in val_loader: 
        files = files.to(device)
        labels = labels.to(device)
        outputs = model(files)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss/len(train_loader):.6f} \t\t Validation Loss: {valid_loss/len(val_loader):.6f} \t\t Accuracy of the model on validation set: {correct/total:.6f}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')


# In[ ]:


# Load Best Performing Model
model = ConvNeuralNet(num_classes)
model.load_state_dict(torch.load('saved_model.pth'))
model = model.to(device)
model.eval()


# In[ ]:


# Predict on test dataset
g = open('test.json')
test = json.load(g)
df_test = pd.DataFrame.from_dict(test, orient='index', columns = ['Prediction'])

y_test_placeholder = random.sample(range(1, 100000), len(df_test['Prediction']))
y_test_placeholder = torch.tensor(y_test_placeholder)

test_dataset = TensorDataset(X_test_tensor, y_test_placeholder)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

p = []

for files, labels in test_loader: 
    files = files.to(device)
    labels = labels.to(device)
    predictions = model(files)
    p.append(torch.argmax(predictions, dim=1).tolist())
                
predictions_test = [item for sublist in p for item in sublist]


# In[ ]:


# Save predictions to JSON

df_test['Prediction'] = predictions_test
df_test['Prediction'] = df_test['Prediction'].astype('str')

predictions_test_dict = dict(df_test.iloc[:, -1])

with open('predictions.json', 'w+') as fp:
    json.dump(predictions_test_dict, fp, indent=4)

