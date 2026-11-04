import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


# Suppress MLflow artifact download warnings
# os.environ["MLFLOW_DISABLE_ARTIFACTS_DOWNLOAD"] = "1"

# Set MLflow Tracking URI & Dagshub integration
MLFLOW_TRACKING_URI = ""

