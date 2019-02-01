# -*- coding: utf-8 -*-
# @Time    : 2018/12/30 11:19
# @Author  : 朝天椒
# @Wechat公众号  : 辣椒哈哈
# @FileName: data_analysis
# @Software: PyCharm


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
sns.set(color_codes=True)


from sqlalchemy import create_engine

import plotly.offline as offline
import plotly.plotly as py
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt

from sklearn import  linear_model

%matplotlib inline
offline.init_notebook_mode(connected=False)

import os
os.listdir('./')

















