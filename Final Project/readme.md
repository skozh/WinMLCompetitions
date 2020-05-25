Hello!

Below you can find a outline of how to reproduce my solution for the Kaggler: Predict Future Sales competition.
If you run into any trouble with the setup/code or have any questions please contact me at shyjukozhisseri@gmail.com

#ARCHIVE CONTENTS
sales-prediction-submitted.ipynb            : original ipynb file - contains original code, EDA, etc
Source/Data/                                : folder contains input data training/test set
Source/Output/                              : folder contains the final output predictions
Source/main.py                              : original code to train and make predictions
Source/funs.py                              : original code for functions used in main.py

#HARDWARE: (The following specs were used to create the original solution)
Ubuntu 16.04 LTS (512 GB boot disk)
n1-standard-16 (16 vCPUs, 60 GB memory)

#SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.5.1

#DATA SETUP 
Copy the input training dataset to the DATA folder.

#DATA PROCESSING
python ./main.py

#MODEL BUILD: 
Currently the build takes in to consideration lags for 4 months and data after month 6.
Change the lags to include more months for training if needed and re-run the training.
