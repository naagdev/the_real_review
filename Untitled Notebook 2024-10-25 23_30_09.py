# Databricks notebook source
import numpy as np
import pandas as pd
import pyspark.pandas as ps
 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from pyspark.sql.types import StructField,IntegerType, StructType,StringType
from pyspark.sql.functions import col

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
 
from pyspark.sql.functions import *

import matplotlib.pyplot as plt

# COMMAND ----------

df = spark.sql('select * from default.all_beauty')

# COMMAND ----------

df.groupBy('rating').agg({'user_id':'count'}).show()

# COMMAND ----------

df_agg = df.groupBy('asin').agg({'rating':'avg','user_id':'count'}
                                ).withColumnRenamed('count(user_id)', 'user_count'
                                                    ).orderBy('user_count', ascending=False)
df_agg.show(1000)
# .orderBy('user_id', ascending=False)

# COMMAND ----------


