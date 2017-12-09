# Sentiment Analysis on Twitter data 

For the course "Pattern Classification and Machine Learning" at EPFL, we worked on sentiment analysis over twitter data. This project was a competition hosted by Kaggle : https://www.kaggle.com/c/epfml17-text
We had at our disposal two sample files of negative and positive labels both containing 100,000 tweet.
Also, we had a complete dataset of 2,500,000 tweets (1,250,000 for each label)
The dataset has been labeled by the presence of  ":)" for positive tweets and ":(" for negative tweets 
you can download the dataset on https://www.kaggle.com/c/epfml17-text/data

## Results

After building the 2 models, we fitted XGBoost over the matrix of probabilities (2 by 200,000) which yield the final result.

In our final model we take just the small sample files of the datasets (200,000 tweets) for the train, to fit our model.

In these 2 models, we mainly used LSTM, Convolutions, MaxPooling layers. We mixed them by changing the seeds and the set of features.
You can see the details of the models on Final/models.
Here are the results for the 2 models and the final result :

| Models       | Accuracy           | Validation Acc |
| -------------|:------------------:|:-------------------:|
| Model 1      | ????????           | ????????            |
| Model 2      | ????????           | ????????            |


After that, we applied XGBoost over the matrix of probabilities which resulted in an accuracy of ???????? and a validation accuracy (submission in kaagle) of ????????.

We were ranked ??st out of ?? teams and scored 0.8402 (private) on kaggle
You can see the leaderboard on : https://www.kaggle.com/c/epfml17-text/leaderboard

We should notice that we didn't use the full dataset (250.000.000) but only the small portion of the dataset which corresponds to 8% of the full datasets. Indeed, we were not able to run our models for the full dataset because it need a high computationnal power. Therefore, we strongly believe that we can really improve our score if we would be able to fit our model with all the dataset using the same models. 

## Files/Folders

The folder contains the followings:

- `packages.txt`: Contains the required packages to run our model
- `features.py`: Contains the details of the building of the feature matrix
- `models.py`: Contains the details of the both models 
- `run.py`: Load the pickled neural network models + fits the obtained results with XGBoost + Creates the Kaggle csv submission
- `preprocess.py`: Preprocesses all the tweets (Cleaning part of the tweets)
- `dico`: This folder contains a dictionnary that will help us for the preprocessing
- `features`: This folder contains the pickled files of the two models for both the train and test set

## How to run the code

- Start by installing the packages in the Final folder :
```
$ pip install -r packages.txt
```
If you are in Mac OS X you can face a problem to install the xgboost package. So, in this case, you can tape the following instructions:

```
$ git clone --recursive https://github.com/dmlc/xgboost
$ cd xgboost
$ cp make/minimum.mk ./config.mk
$ make -j4
$ cd python-package
$ sudo python setup.py install

```
If you have a problem with the keras package, you should just verify that the tensorflow is properly installed.


- To run the final model :

We stored all the features of the two models in the folder features.
To run the models using the pickled features we provide :
 
```
$ python3 run.py 
```

This will yield our Kaggle prediction that scored 0.84020.

We used here just the small sample files of datasets ( 200,000 tweets). We also added the bi-grams and we fitted the XGBoost over the matrix of probabilities (2 by 200,000).

However, if you want to run the models from the start, please follow these steps :

- run the models in the file `models.py` and dump the features, which will save the pickled files of the two models in the folder `features`. You can also run just one model by commenting the other.
- You can also modify the value of the arguments of the function `dumpFeatures` in the file `models.py` to run a personalized model.
- After dumping all the features, load them and run XGBoost on the probability matrix (by means of `run.py` ).


## Authors

- Mohammed Reda Bousseta : mohammed.bousseta@epfl.ch
- Ismail Bali : ismail.bali@epfl.ch
