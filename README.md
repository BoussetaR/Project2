# Text Sentiment Classification

This project is part of the course "Machine Learning" at EPFL, we choose to work on the Text Sentiment Classification. This project was a competition hosted by Kaggle and this is the link https://www.kaggle.com/c/epfml17-text, you can see our score and our ranking via this link, our team name is "Mario", our final and best score on kaagle is: 0.8588.

# Prerequisites

To start the project, we should understand the data. So, we had two sample files of negative and positive labels both containing 100,000 tweets. In addition to that, we had a full dataset of 2,500,000 tweets ( 1,250,000 for each label).
The dataset has been labeled by the presence of  ":)" for positive tweets and ":(" for negative tweets.
You can download the dataset on https://www.kaggle.com/c/epfml17-text/data.

We should notice that we didn't use all the full dataset (250.000.000) but only a portion of 22% of the full dataset which corresponds to: 550.000 tweets. Indeed, we were not able to run our models for the full dataset because it needs a high computationnal power. Therefore, we strongly believe that we can really improve our score if we would be able to fit our model with all the dataset using the same models.

## Description of Files and Folders

The folder contains the followings:

- `packages.txt`: Contains the required packages to run our model
- `dico_preprocess.txt`: Contains the dictionnary that we will use to correct the mistakes in the preprocessing
- `building_features.py`: Contains the function which build the feature matrix
- `models.py`: Contains the details of the both models and create the pickled_features
- `pickled_features`: This folder contains the pickled files of the two models for both the train and test set
- `run.py`: Load the pickled models and fits the obtained results with XGBoost and finaly Creates the csv file for submission

## Running the code

- You should start by downloading the data from the link and put them in the folder.
- Install the packages that we need to run our models :
```
$ pip install -r packages.txt
```
You can face a problem to install the xgboost package. So, in this case, you can tape the following instructions:

```
$ git clone --recursive https://github.com/dmlc/xgboost
$ cd xgboost
$ cp make/minimum.mk ./config.mk
$ make -j4
$ cd python-package
$ sudo python setup.py install

```
If you have a problem with the keras package, you should just verify that the tensorflow is properly installed.


- To run the final model with the pickled features, you should just run `run.py`, it will use the pickled files that we stores in the corresponding folder. 
 
```
$ python3 run.py 
```

This will create a csv file that we use for the Kaagle submission. This yield to our best score in Kaagle: 0.8588.


However, if you want to run the models from the beginning, you should follow these steps :

- run the models in the file `models.py` which will save the pickled files of the two models in the folder `pickled_features`.
- You can also modify the value of the arguments of the function `build_features` in the file `models.py` to run a personalized model.
- After this, we just run the file `run.py` to create the csv file that we wil submit in Kaagle to see the score.

## Authors

- Mohammed Reda Bousseta : mohammed.bousseta@epfl.ch
- Ismail Bali : ismail.bali@epfl.ch
