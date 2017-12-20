# Text Sentiment Classification

This project is part of the course "Machine Learning" at EPFL, we choose to work on the Text Sentiment Classification. This project was a competition hosted by Kaggle and this is the link https://www.kaggle.com/c/epfml17-text, you can see our score and our ranking via this link, our team name is "Mario", our final and best score on kaagle is: 0.8588.

# Prerequisites

To start the project, we should understand the data. So, we had two sample files of negative and positive labels both containing 100,000 tweets. In addition to that, we had a full dataset of 2,500,000 tweets ( 1,250,000 for each label).
The dataset has been labeled by the presence of  ":)" for positive tweets and ":(" for negative tweets.
You can download the dataset on https://www.kaggle.com/c/epfml17-text/data.

We should notice that we didn't use all the full dataset (250.000.000) but only a portion of 22% of the full dataset which corresponds to: 550.000 tweets. Indeed, we were not able to run our models for the full dataset because it need a high computationnal power. Therefore, we strongly believe that we can really improve our score if we would be able to fit our model with all the dataset using the same models. 

## Description of Files and Folders

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
