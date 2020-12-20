
import keras
import matplotlib
import nltk
import pandas as pd
import numpy as np
import re
import codecs
import xgboost
from pasta.augment import inline

from sklearn.model_selection import train_test_split

from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from sklearn.svm import SVC
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

from keras.models import Sequential
from keras import layers

from transformers import BertTokenizer
import tensorflow as tf

import os
import pandas as pd
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import torch
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import  tqdm_notebook

from prepare_data import *
from regressiones import *
from neural_network import *
from bert import *