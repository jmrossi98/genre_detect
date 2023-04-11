<img src="/images/title.png" alt="Title" width="1200">

<br />

A music classification tool that uses deep learning to predict the genre of a given audio file.

This model is a Long Short-Term Memory Recurrent Neural Network trained on the [GTZAN Genre Collection Dataset](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection).

It'll classify the audio file into one of 10 categories: Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae or Rock.

<br />

## Installation

Create a local instance by running
` git clone https://github.com/jmrossi98/genre_detect.git `

Install all dependencies by running
`pip install -e .`

<br />

## Usage

All you need is to specify the path of an audio file you'd like to classify.

#### --path

> No other arguments will have the tool use the RNN-LSTM model I trained with the GTZAN dataset.

Example: <br />
`python genre_detect.py --path C:\Users\jake\Downloads\shook_ones_pt2.mp3`

<br />

### Optional Arguments

If you want to preprocess new data and/or build a new model you can add these arguments:

#### --name

> Specify the model name you want to predict on.

Example: <br />
`python genre_detect.py --name rnn_genre_classifier_new --path C:\Users\jake\Downloads\shook_ones_pt2.mp3`

<br />

#### --preprocess

> The path to raw data that can be preprocessed for the model to use.

Example: <br />
`python genre_detect.py --preprocess data\archive\Data\genres_original`

<br />

#### --build

> Rebuild a model to train and save under specified name in models folder. You can edit the model at src\build_model. Default model will be overwritten if name isn't specified.

Example: <br />
`python genre_detect.py --build --name rnn_genre_classifier_new`

<br />

## Accuracy/Error Eval
Test accuracy plot of the training process to find the default model (models\rnn_genre_classifier.h5). This was selected as the model with the best validation accuracy throughout all epochs.

<br />

<img src="/images/model_eval.png" alt="ModelEval" width="1200">

