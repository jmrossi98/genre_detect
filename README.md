# Genre Detect

A tool that predicts the genre of a given audio file using a Long Short-Term Memory Recurrent Neural Network.

This model will classify the audio file into one of 10 categories: Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae or Rock.

## Installation

Create a local instance by running
` git clone https://github.com/jmrossi98/genre_detect.git `

Install all dependencies by running
`pip install -e .`

## Usage

### Required

All you need is to specify the path of an audio file you'd like to classify.

#### --path

> No other arguments will have the tool use the RNN-LSTM model I trained with the GTZAN dataset.

Example:
`python genre_detect.py --path C:\Users\jake\Downloads\shook_ones_pt2.mp3`


### Optional

If you want to preprocess new data and/or build a new model you can add these arguments:

#### --preprocess

> The path to raw data that can be preprocessed for the model to use.

Example:
`python genre_detect.py --preprocess data\archive\Data\genres_original`

#### --build and --name

> Rebuild a model to train and save under specified name in models folder. You can edit the model at src\build_model. Default model will be overwritten if name isn't specified.

Example:
`python genre_detect.py --build --name rnn_genre_classifier_new`


## Model
Test accuracy plot of the model I trained and saved at models\rnn_genre_classifier.h5
> <img src="/images/model_eval.png" alt="ModelEval" width="1200">

