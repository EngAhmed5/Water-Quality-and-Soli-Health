import joblib
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import webbrowser
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  ConfusionMatrixDisplay, classification_report
import scienceplots
plt.style.use(['science','nature'])
plt.rcParams["text.usetex"] = False   # <-- FIX

import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE=42
DATAPATH = r"D:\Ahmed\Study\Graduation project\codes\data2\data_core.csv"