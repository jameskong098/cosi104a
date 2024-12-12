#!/usr/bin/env python
# coding: utf-8

# # Kaggle Competition : Child Mind Institute — Problematic Internet Use
# ### Relating Physical Activity to Problematic Internet Use
# ### Competition Link : <https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview>
# ### Compeition Dataset Link : <https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data>

# 
# ## train.csv and test.csv comprises measurements from a variety of instruments
# 
# - **Demographics** - Information about age and sex of participants.
# - **Internet Use** - Number of hours of using computer/internet per day.
# - **Children's Global Assessment Scale** - Numeric scale used by mental health clinicians to rate the general functioning of youths under the age of 18.
# - **Physical Measures** - Collection of blood pressure, heart rate, height, weight and waist, and hip measurements.
# - **FitnessGram Vitals and Treadmill** - Measurements of cardiovascular fitness assessed using the NHANES treadmill protocol.
# - **FitnessGram Child** - Health related physical fitness assessment measuring five different parameters including aerobic capacity, muscular strength, muscular endurance, flexibility, and body composition.
# - **Bio-electric Impedance Analysis** - Measure of key body composition elements, including BMI, fat, muscle, and water content.
# - **Physical Activity Questionnaire** - Information about children's participation in vigorous activities over the last 7 days.
# - **Sleep Disturbance Scale** - Scale to categorize sleep disorders in children.
# - **Actigraphy** - Objective measure of ecological physical activity through a research-grade biotracker.
# - **Parent-Child Internet Addiction Test** - 20-item scale that measures characteristics and behaviors associated with compulsive use of the Internet including compulsivity, escapism, and dependency.
# 
# - **Note** in particular the field **PCIAT-PCIAT_Total.** The target **sii** for this competition is derived from this field as described in the data dictionary: **0 for None, 1 for Mild, 2 for Moderate, and 3 for Severe.** Additionally, each participant has been assigned a unique identifier id.
# 
# ### Actigraphy Files and Field Descriptions
# During their participation in the HBN study, some participants were given an accelerometer to wear for up to 30 days continually while at home and going about their regular daily lives.
# 
# - **series_{train|test}.parquet/id={id}** - Series to be used as training data, partitioned by id. Each series is a continuous recording of accelerometer data for a single subject spanning many days.
# 
# - **id** - The patient identifier corresponding to the id field in train/test.csv.
# - **step** - An integer timestep for each observation within a series.
# - **X, Y, Z** - Measure of acceleration, in g, experienced by the wrist-worn watch along each standard axis.
# - **enmo** - As calculated and described by the wristpy package, ENMO is the Euclidean Norm Minus One of all accelerometer signals (along each of the x-, y-, and z-axis, measured in g-force) with negative values rounded to zero. Zero values are indicative of periods of no motion. While no standard measure of acceleration exists in this space, this is one of the several commonly computed features.
# - **anglez** - As calculated and described by the wristpy package, Angle-Z is a metric derived from individual accelerometer components and refers to the angle of the arm relative to the horizontal plane.
# - **non-wear_flag** - A flag (0: watch is being worn, 1: the watch is not worn) to help determine periods when the watch has been removed, based on the GGIR definition, which uses the standard deviation and range of the accelerometer data.
# - **light** - Measure of ambient light in lux. See ​​here for details.
# - **battery_voltage** - A measure of the battery voltage in mV.
# - **time_of_day** - Time of day representing the start of a 5s window that the data has been sampled over, with format **%H:%M:%S.%9f.**
# - **weekday** - The day of the week, coded as an integer with 1 being Monday and 7 being Sunday.
# - **quarter** - The quarter of the year, an integer from 1 to 4.
# - **relative_date_PCIAT** - The number of days (integer) since the PCIAT test was administered (negative days indicate that the actigraphy data has been collected before the test was administered).

# # Install library

# In[1]:

# # Config

# In[2]:


class CFG:
    
    USE_GPU= False
    test =0.2
    normalize = False#True
    overSampling = True # handling imbalance class
    crossValidate = True#True # Support cross Validation 
    nFold = 5#4 # 8  No remainder  for oversampling 

    #Ensemble Learning Algorithm Select
    USE_ENSEMBLE = True #False #True
    # USE_VOTING = True
    # USE_WEIGHT = False
    # USE_STACK = False
    

    # XGBoost
    xgbEstimate = 350 #300 #250 #150 #800 #500 #200 # 250 , 300 
    xgbDepth = 3 #5 #7 #5 #6, low value avoid overfit
    xgbLR = 0.03 #0.05 # 0.08
    xgbEarlyStop= 8 #10 #20
    colsample_bytree = 0.5 #0.8

    # Tabnet model hyperparameter 
    maxEpochs = 1000 #500 #300 #150 #100#200 #300
    patience = 30 #20 #50 #50 #10 #8 #5
    n_d = 64#32 #8
    n_a = 64#32 # 8
    n_steps = 5 # 3
    mask_type = "sparsemax" #"entmax" # "sparsemax"
    learningRate = 1e-2 #1e-3 # 1e-2

    # lightGBM
    lgbmEstimators = 200
    
    trainSeriesParquet = "series_train.parquet"
    testSeriesParquet = "series_test.parquet"
    rootDir ="/kaggle/input/child-mind-institute-problematic-internet-use/"
    dataDictionary = "/kaggle/input/child-mind-institute-problematic-internet-use/data_dictionary.csv"
    trainDataFile= "/kaggle/input/child-mind-institute-problematic-internet-use/train.csv"
    testDataFile = "/kaggle/input/child-mind-institute-problematic-internet-use/test.csv"
    samplesub = "/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv"
    


# In[3]:


import os, gc, time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from IPython.display import Markdown
import ctypes
import seaborn as sns
from tqdm import tqdm

import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.callbacks import Callback


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (train_test_split, KFold, GridSearchCV, StratifiedKFold,
                                    cross_val_score)
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# over Sample 
from imblearn.over_sampling import SMOTE

from sklearn.metrics import ( mean_absolute_error, 
                             mean_squared_error, 
                             r2_score, 
                             cohen_kappa_score)

from sklearn.model_selection import (train_test_split, 
                                     KFold, 
                                     GridSearchCV, 
                                     StratifiedKFold)


# import torch
# ML library 
from xgboost import XGBRFRegressor, XGBClassifier# plot_importance, plot_tree
import xgboost
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm
from catboost import CatBoostRegressor, CatBoostClassifier
import catboost
from sklearn.ensemble import (RandomForestClassifier ,VotingClassifier )

# multiple processing lib
from joblib import Parallel , delayed
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


from warnings import simplefilter 
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:





# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# # Load dataset

# ### Load meta Data 
# #### help us to understand the data feature

# In[5]:


get_ipython().run_cell_magic('time', '', 'dataDictDF = pd.read_csv(CFG.dataDictionary)\ndataDictDF\n')


# In[6]:


pd.set_option('display.max_rows', 50)


# In[7]:


get_ipython().run_cell_magic('time', '', 'trainDataDF = pd.read_csv(CFG.trainDataFile)\ntrainDataDF\n')


# In[8]:


trainDataDF.describe()


# In[9]:


get_ipython().run_cell_magic('time', '', 'testDataDF = pd.read_csv(CFG.testDataFile)\ntestDataDF\n')


# In[10]:


testDataDF.shape


# In[ ]:





# In[11]:


submit = pd.read_csv(CFG.samplesub)
submit


# In[12]:


def cleanMemory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()


# In[13]:


cleanMemory()


# ## Print Columns Values 

# In[14]:


def printAllcolumnsValue(df, showAll=True):
    for col in df.columns:
        if showAll :
            print(f"{col} : {df[col].unique()}") # print unique value
        else: # only print catergory column
            if df[col].dtype == "object":
                print(f"{col} : {df[col].unique()}") # print unique value


# In[15]:


def printSerieUnquieValue(df):
    print(f" {df.unique()}")


# In[16]:


printSerieUnquieValue(trainDataDF["Basic_Demos-Age"])


# In[17]:


printSerieUnquieValue(trainDataDF["Physical-Waist_Circumference"])


# In[18]:


printSerieUnquieValue(trainDataDF["Physical-Height"])


# In[19]:


printSerieUnquieValue(trainDataDF["BIA-BIA_Fat"])


# ### Get Feature 

# In[20]:


for item in trainDataDF.columns:
    print(item)


# In[21]:


noColumnInTest = [ item for item in trainDataDF.columns  if item not in testDataDF.columns]
noColumnInTest


# In[22]:


featureCols = trainDataDF.columns.tolist()
featureCols.remove("id")
featureCols , len(featureCols)


# In[ ]:





# ## load Time serise data

# In[23]:


# read one of time series data
trainSerise1 = pd.read_parquet("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet/id=00115b9f/part-0.parquet")
trainSerise1


# In[24]:


trainSerise1.drop("step", axis=1, inplace=True)
trainSerise1


# In[25]:


sensorFeature =trainSerise1.columns.to_list()
sensorFeature , len(sensorFeature)


# In[26]:


# get Statistic Feature list
statFeat = trainSerise1.describe().index.to_list() 
statFeat, len(statFeat)


# # generate senor feature statistic column list

# In[27]:


# generate senor feature statistic column list
sensorStatCol = []
for sta in statFeat: # start statist row 
    # print(sta)
    for item in sensorFeature:
        # print(item)
        sensorStatCol.append(f"{item}_{sta}")
        
sensorStatCol, len(sensorStatCol)


# In[ ]:





# In[28]:


# Result 12(paramter) x 8 (statistic) = 96 (Data vector) summary (statistic for each type of sensor feature)
trainSerise1.describe().values.reshape(-1)


# In[29]:


def loadParquetFile(directory, fileName):
    """
    read parquet file 
    """
    path = os.path.join(directory, fileName, "part-0.parquet")
    df = pd.read_parquet(path)
    df.drop("step", axis=1, inplace=True) # drop step column
    statDF = df.describe().values.reshape(-1)
    return statDF, fileName.split("=")[1] # get ids
    

def loadTimeSeriesData(directory):
    """
    input : root direction
    """
#     print("DIR :", directory)
    filesIds = os.listdir(directory) # get list of folder name (files ids)
#     print(filesIds)
    with ThreadPoolExecutor() as executor:
#     with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: loadParquetFile(directory, fname), filesIds), total=len(filesIds)))
#     print(results)
    statistic, ids = zip(*results) # pack into Statistic and Ids 
    # create new dataframe with n statistic sensor data
#     data = pd.DataFrame(statistic, columns=[f"stat_{i}" for i in range(len(statistic[0]))])
    data = pd.DataFrame(statistic, columns=sensorStatCol)
    data["id"] = ids # add ids into dataframe
#     print(data)
    
    return data


# In[30]:


get_ipython().run_cell_magic('time', '', 'trainTSData = loadTimeSeriesData((CFG.rootDir + CFG.trainSeriesParquet))\ntestTSData = loadTimeSeriesData((CFG.rootDir + CFG.testSeriesParquet))\n')


# In[31]:


trainTSData.head()


# In[32]:


trainTSData.info()


# In[33]:


trainTSData.isna().sum()


# In[34]:


testTSData


# ## Combine Time series data into dataset

# In[35]:


combinedTrainDF =  pd.merge(trainDataDF, trainTSData, how="left", on='id')
combinedTestDF = pd.merge(testDataDF, testTSData, how="left", on='id')


# In[36]:


combinedTrainDF


# In[37]:


combinedTestDF


# In[38]:


combinedTrainDF = combinedTrainDF.drop("id", axis=1)
combinedTestDF = combinedTestDF.drop("id", axis=1)


# In[39]:


combinedTrainDF


# ## Update the feature column list after combined time series featureS

# In[40]:


featureCols = combinedTrainDF.columns.tolist()
featureCols,  len(featureCols)


# In[41]:


combinedTestDF


# ## Data cleaning

# In[42]:


combinedTrainDF["sii"].isnull().sum() # count number of Nan row in sii target column


# In[43]:


combinedTrainDF = combinedTrainDF.dropna(subset="sii")
combinedTrainDF


# In[44]:


combinedTrainDF["sii"].isnull().sum() # check Nan in sii column


# ## find caterogy column

# In[45]:


#find 
printAllcolumnsValue(combinedTrainDF, showAll= False)


# In[46]:


# find which column if catargory
columnsCaterogy = []
for col in combinedTrainDF.columns:
    if combinedTrainDF[col].dtype == "object":
        columnsCaterogy.append(col)


# In[47]:


columnsCaterogy , len(columnsCaterogy)


# In[48]:


# find which colum if numunic
columnsNum = []
for col in combinedTrainDF.columns:
    if combinedTrainDF[col].dtype != "object" and col !="sii":
        columnsNum.append(col)


# In[49]:


len(columnsNum)


# In[50]:


columnsTestCaterogy = []
for col in combinedTestDF.columns:
    if combinedTestDF[col].dtype == "object":
        columnsTestCaterogy.append(col)


# In[51]:


columnsTestCaterogy , len(columnsTestCaterogy)


# In[52]:


# find which colum if numunic
columnsTestNum = []
for col in combinedTestDF.columns:
    if combinedTestDF[col].dtype != "object" and col !="sii":
        columnsTestNum.append(col)


# In[53]:


len(columnsTestNum)


# ## Seem the Training dataset column more than test(submi) dataset, the training dataset should same feature size with test(submit) feature size 
# ### select feature for training dateset same with testing dataset

# In[ ]:





# In[54]:


tempCol =columnsTestCaterogy + columnsTestNum +["sii"]
tempCol2 = columnsTestCaterogy + columnsTestNum
len(tempCol) , len(tempCol2)


# In[55]:


tempCol2


# In[56]:


# select column for match both training and testing feature
combinedTrainDF =combinedTrainDF[tempCol]
combinedTestDF = combinedTestDF[tempCol2]


# In[57]:


combinedTrainDF.describe() # final train dataset feature is 155 with label


# ## Other column Nan count

# In[58]:


combinedTrainDF.isnull().sum()


# ## cleaning/handling other column Nan value

# In[59]:


combinedTrainDF.isnull().sum()


# In[60]:


combinedTestDF.isnull().sum()


# In[61]:


combinedTestDF.columns


# In[62]:


# fill the Nan (missing) in for category
for c in columnsTestCaterogy:
    combinedTrainDF[c] = combinedTrainDF[c].fillna('missing') # fill na for missing caterogy
    combinedTestDF[c] = combinedTestDF[c].fillna("missing") # fill na for missing caterogy
    


# In[63]:


# fill the Nan numberic with median value 
for c in columnsTestNum:
    combinedTrainDF[c] = combinedTrainDF[c].fillna(combinedTrainDF[c].median())
    combinedTestDF[c] = combinedTestDF[c].fillna(combinedTestDF[c].median())


# In[64]:


combinedTrainDF.isnull().sum() # check is any null in column


# In[65]:


combinedTestDF.isnull().sum()


# ## Encode the Cateroy column into numberical 

# In[66]:


# define Category dictionary for encode 
catDic = {"missing": 0, "Spring": 1, "Summer": 2, "Fall": 3, "Winter" : 4}


# In[67]:


# labelEncoder = LabelEncoder()
# for c in columnsTestCaterogy:
#     combinedTrainDF[c] = labelEncoder.fit_transform(combinedTrainDF[c]).astype(int)
#     combinedTestDF[c] = labelEncoder.fit_transform(combinedTestDF[c]).astype(int)
for c in columnsTestCaterogy:
    combinedTrainDF[c] = combinedTrainDF[c].map(catDic)
    combinedTestDF[c] =combinedTestDF[c].map(catDic)


# ## find incorrect value in weight

# In[68]:


pd.set_option('display.max_rows', None)


# In[69]:


combinedTrainDF[combinedTrainDF['Physical-Weight'] ==0]


# In[70]:


combinedTrainDF['Physical-Weight'].replace(0, combinedTrainDF['Physical-Weight'].median() , 
                                           inplace=True )


# In[71]:


combinedTrainDF[combinedTrainDF['Physical-Weight'] == 0]


# # find incorrect value in BMI

# In[72]:


# 703  *  75.8/ (60.50**2)


# In[73]:


combinedTrainDF[combinedTrainDF["Physical-BMI"]== 0]


# In[74]:


combinedTrainDF["Physical-BMI"].replace(0, combinedTrainDF["Physical-BMI"].median() , inplace=True)


# In[75]:


combinedTrainDF[combinedTrainDF["Physical-BMI"]== 0]


# In[76]:


pd.set_option('display.max_rows', 50)


# # Feature Engineering, create new feature for improve model training/prediction

# # Feature Engineering for generate new column

# # Define Complex Health Analysis function

# In[77]:


def getBMIInfo(df):
    """
    classify Weight caterogy by BMI reference from World Health Organization  
    """
    # print(df)
    bmi = df["Physical-BMI"]
    age = df["Basic_Demos-Age"]
    gender = df["Basic_Demos-Sex"]
    bodyweightClass = 0
    if df["Basic_Demos-Sex"] == 0: # for female
        if age < 8 :
            adaptThreshold1 = 12.0
            adaptThreshold2 = 13.0
            adaptThreshold3 = 17.7
            adaptThreshold4 = 0.5*(age-5) + 19.0 
            if bmi < adaptThreshold1:
                bodyweightClass = 0 # Severe thinness
            elif bmi <adaptThreshold2:
                bodyweightClass = 1 # thiness
            elif bmi <adaptThreshold3:
                bodyweightClass = 2 # normal 
            elif bmi <adaptThreshold4:
                 bodyweightClass = 3 # Overweight 
            else :
                 bodyweightClass = 4 # Obesity

            return bodyweightClass
            
        elif age < 15:
            # apply linear regression 
            adaptThreshold1 = 0.357 * (age-8) + 12.0
            adaptThreshold2 = 0.429 * (age-8) + 13.0
            adaptThreshold3 = 0.828 * (age-8) + 17.7
            adaptThreshold4 = 1.1*(age-8) + 20.5 
            if bmi < adaptThreshold1:
                bodyweightClass = 0 # Severe thinness
            elif bmi <adaptThreshold2:
                bodyweightClass = 1 # thiness
            elif bmi <adaptThreshold3:
                bodyweightClass = 2 # normal 
            elif bmi <adaptThreshold4:
                 bodyweightClass = 3 # Overweight 
            else :
                 bodyweightClass = 4 # Obesity

            return bodyweightClass
        
        elif age < 19:
            # apply linear regression 
            adaptThreshold1 = 14.7
            adaptThreshold2 = 16.5
            adaptThreshold3 = 0.5 * (age-15) + 23.5
            adaptThreshold4 = 0.325 *(age-15) + 28.2 
            if bmi < adaptThreshold1:
                bodyweightClass = 0 # Severe thinness
            elif bmi <adaptThreshold2:
                bodyweightClass = 1 # thiness
            elif bmi <adaptThreshold3:
                bodyweightClass = 2 # normal 
            elif bmi <adaptThreshold4:
                 bodyweightClass = 3 # Overweight 
            else :
                 bodyweightClass = 4 # Obesity

            return bodyweightClass
        else: 
            #use adult
            if bmi < 18.5: 
                bodyweightClass = 0 # thiness 
            elif bmi < 24.9:
                bodyweightClass = 2 # noraml 
            elif bmi < 29.9:
                 bodyweightClass = 3 # Overweight 
            else:
                bodyweightClass = 4 # Obesity

            return bodyweightClass


    else: # for male
        if age < 8 :
            adaptThreshold1 = 12.5
            adaptThreshold2 = 13.2
            adaptThreshold3 = 0.266 *(age-5) + 16.7
            adaptThreshold4 = 0.5*(age-5) + 18.2 
            if bmi < adaptThreshold1:
                bodyweightClass = 0 # Severe thinness
            elif bmi <adaptThreshold2:
                bodyweightClass = 1 # thiness
            elif bmi <adaptThreshold3:
                bodyweightClass = 2 # normal 
            elif bmi <adaptThreshold4:
                 bodyweightClass = 3 # Overweight 
            else :
                 bodyweightClass = 4 # Obesity

            return bodyweightClass
            
        elif age < 15:
            # apply linear regression 
            adaptThreshold1 = 0.314 * (age-8) + 12.5
            adaptThreshold2 = 0.4 * (age-8) + 13.2
            adaptThreshold3 = 0.742 * (age-8) + 17.5
            adaptThreshold4 = 1.04 *(age-8) + 19.7 
            if bmi < adaptThreshold1:
                bodyweightClass = 0 # Severe thinness
            elif bmi <adaptThreshold2:
                bodyweightClass = 1 # thiness
            elif bmi <adaptThreshold3:
                bodyweightClass = 2 # normal 
            elif bmi <adaptThreshold4:
                 bodyweightClass = 3 # Overweight 
            else :
                 bodyweightClass = 4 # Obesity

            return bodyweightClass
        
        elif age < 19:
            # apply linear regression 
            adaptThreshold1 = 15.7
            adaptThreshold2 = 0.375 * (age-15) + 16.0
            adaptThreshold3 = 0.7 * (age-15) + 22.7
            adaptThreshold4 = 0.675 *(age-15) + 27.0
            if bmi < adaptThreshold1:
                bodyweightClass = 0 # Severe thinness
            elif bmi <adaptThreshold2:
                bodyweightClass = 1 # thiness
            elif bmi <adaptThreshold3:
                bodyweightClass = 2 # normal 
            elif bmi <adaptThreshold4:
                 bodyweightClass = 3 # Overweight 
            else :
                 bodyweightClass = 4 # Obesity

            return bodyweightClass
        else: 
            #use adult
            if bmi < 18.5: 
                bodyweightClass = 0 # thiness 
            elif bmi < 24.9:
                bodyweightClass = 2 # noraml 
            elif bmi < 29.9:
                 bodyweightClass = 3 # Overweight 
            else:
                bodyweightClass = 4 # Obesity
            return bodyweightClass


# In[78]:


def getWaistHeightInfo(df):
    age = df["Basic_Demos-Age"]
    gender = df["Basic_Demos-Sex"]
    ratio = df["Physical-Waist_Circumference"] / df["Physical-Height"]
    ratioClass = 0
    if age < 15:
        if ratio < 0.34:
            ratioClass =0 # Extremely Slim
        elif ratio < 0.45:
            ratioClass  = 1 # slim
        elif ratio < 0.51:
            ratioClass = 2 # Healthy
        elif ratio < 0.63:
            ratioClass = 3 # over weight
        else:
            ratioClass = 5 # Obese
        return ratioClass 

    else : # for adult 
        if gender == 0: # for female
            if ratio < 0.34:
                ratioClass =0 # Extremely Slim
            elif ratio < 0.41:
                ratioClass  = 1 # slim
            elif ratio < 0.48:
                ratioClass = 2 # Healthy
            elif ratio < 0.53:
                ratioClass = 3 # over weight
            elif ratio < 0.57:
                ratioClass = 4 # Very Overweight
            else:
                ratioClass = 5 # Obese
            return ratioClass
        else : # for male
            if ratio < 0.34:
                ratioClass =0 # Extremely Slim
            elif ratio < 0.42:
                ratioClass  = 1 # slim
            elif ratio < 0.52:
                ratioClass = 2 # Healthy
            elif ratio < 0.57:
                ratioClass = 3 # over weight
            elif ratio < 0.62:
                ratioClass = 4 # Very Overweight
            else:
                ratioClass = 5 # Obese
            return ratioClass


# In[79]:


def getBodyFatPercentInfo(df):
    """
    classify the user health status by body fat percentage
    """
    gender = df["Basic_Demos-Sex"]
    fatPercent = df["BIA-BIA_Fat"]
    fatclass =0
    if gender == 0: # for female
        if fatPercent < 13:  
            fatclass = 0 # Essential Fat
        elif fatPercent < 20:
            fatclass = 1 # Athletes
        elif fatPercent < 24:
            fatclass = 2 # Fitness
        elif fatPercent < 31:
            fatclass = 3 # acceptable 
        else:
            fatclass = 4 # acceptable 
        return fatclass
    else: # for male
        if fatPercent < 5:  
            fatclass = 0 # Essential Fat
        elif fatPercent < 13:
            fatclass = 1 # Athletes
        elif fatPercent < 17:
            fatclass = 2 # Fitness
        elif fatPercent < 24:
            fatclass = 3 # acceptable 
        else:
            fatclass = 4 # acceptable
        return fatclass


# In[80]:


combinedTrainDF.describe()


# ### Extract Feature

# In[81]:


def featureEngineering(df):
    df["BMI-AGE"] = df["Physical-BMI"] * df["Basic_Demos-Age"]
    df["BMI-Classify"]= df.apply(getBMIInfo, axis=1)
    df["Waist-Height-Ratio"] = df["Physical-Waist_Circumference"] / df["Physical-Height"]
    df["Waist-Height-Ratio-Classify"] = df.apply(getWaistHeightInfo, axis=1) 
    df['Internet-Hours-Age'] = df['PreInt_EduHx-computerinternet_hoursday'] * df['Basic_Demos-Age']
    df['BMI-Internet-Hours'] = df['Physical-BMI'] * df['PreInt_EduHx-computerinternet_hoursday']
    df["FATPercent-Classify"]= df.apply(getBodyFatPercentInfo, axis= 1)
    df['ICW_TBW'] = df['BIA-BIA_ICW'] / df['BIA-BIA_TBW']
    df['FFMI_BFP'] = df['BIA-BIA_FFMI'] / df['BIA-BIA_Fat'] #  
    df['FMI_BFP'] = df['BIA-BIA_FMI'] / df['BIA-BIA_Fat']
    df['LST_TBW'] = df['BIA-BIA_LST'] / df['BIA-BIA_TBW']
    df['BFP_BMR'] = df['BIA-BIA_Fat'] * df['BIA-BIA_BMR']
    df['BFP_DEE'] = df['BIA-BIA_Fat'] * df['BIA-BIA_DEE']
    df['BMR_Weight'] = df['BIA-BIA_BMR'] / df['Physical-Weight']
    df['BMR_Weight'] =df['BMR_Weight'].fillna(0)
    df['DEE_Weight'] = df['BIA-BIA_DEE'] / df['Physical-Weight']
    df['DEE_Weight'] = df['DEE_Weight'].fillna(0)
    df['SMM_Height'] = df['BIA-BIA_SMM'] / df['Physical-Height']
    # Feature 3
    df['Muscle_to_Fat'] = df['BIA-BIA_SMM'] / df['BIA-BIA_FMI']
    df['Hydration_Status'] = df['BIA-BIA_TBW'] / df['Physical-Weight']
    df['BFP_BMI'] = df['BIA-BIA_Fat'] / df['BIA-BIA_BMI']
    
    return df
    


# # 

# In[ ]:





# In[ ]:





# In[ ]:





# In[82]:


printSerieUnquieValue(combinedTrainDF["BIA-BIA_DEE"])


# In[83]:


printSerieUnquieValue(combinedTrainDF["BIA-BIA_Fat"])


# In[84]:


# printSerieUnquieValue(combinedTrainDF['Physical-Weight'])


# In[85]:


combinedTestDF= featureEngineering(combinedTestDF)


# In[86]:


combinedTrainDF = featureEngineering(combinedTrainDF)


# In[87]:


# combinedTestDF["BMI-Classify"] = combinedTestDF.apply(getBMIInfo, axis=1)
combinedTrainDF


# In[88]:


combinedTestDF


# In[89]:


printAllcolumnsValue(combinedTrainDF, showAll= False)


# In[90]:


printAllcolumnsValue(combinedTestDF, showAll= False)


# # update Festure columns list

# In[91]:


tempCol = combinedTrainDF.columns.tolist()
tempCol,  len(tempCol)


# In[92]:


tempCol2 = combinedTestDF.columns.tolist()
tempCol2,  len(tempCol2)


# In[93]:


# combinedTestDF[columnsTestCaterogy[0]].values


# In[94]:


printAllcolumnsValue(combinedTrainDF, showAll= False) # 


# In[95]:


printAllcolumnsValue(combinedTestDF, showAll= False) # 


# In[96]:


combinedTrainDF[columnsTestCaterogy[0]].value_counts() # check Encoded caterogy value


# In[97]:


combinedTestDF[columnsTestCaterogy[0]].value_counts() # check Encoded caterogy value


# In[98]:


combinedTrainDF.isnull().sum()


# In[99]:


combinedTestDF.isnull().sum()


# ## seem The Dataset is very imbalance

# In[100]:


combinedTrainDF["sii"].value_counts().plot(kind="bar", title="sii target catergory Distribution");


# In[101]:


combinedTrainDF.describe()


# In[102]:


combinedTestDF.describe()


# # Train/Validation Split
# ## K-fold Cross Vailation

# In[103]:


xFeature = combinedTrainDF.drop("sii", axis=1)
yLabel = combinedTrainDF["sii"]


# In[104]:


xFeature.shape  , yLabel.shape


# In[105]:


# Handling imbalance with SMOTE
if CFG.overSampling:
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(xFeature, yLabel)
    print(len(X_resampled))
    xFeature = X_resampled
    yLabel = y_resampled




# In[106]:


from sklearn.utils import compute_sample_weight, class_weight


# In[107]:


xFeature.shape


# In[108]:


xFeature


# In[109]:


# scale 
if CFG.normalize:
    scale = StandardScaler()
    scaledXFeature = scale.fit_transform(xFeature)
    xFeature = scaledXFeature
    scaledTestDF    = scale.fit_transform(combinedTestDF)
    combinedTestDF = scaledTestDF


# In[110]:


xFeature


# In[111]:


combinedTestDF


# In[112]:


if CFG.crossValidate:
    straKFold = StratifiedKFold(n_splits=CFG.nFold, random_state=42, shuffle=True)
else:
    X_train, X_test, y_train, y_test = train_test_split(xFeature, yLabel, test_size=CFG.test, 
                                                    random_state=42, stratify=yLabel)
    print(X_train.shape)
    print(X_test.shape)  
    print(y_train.shape)
    print(y_test.shape)
    # Compute sample_weight using compute_sample_weight
    sampleWeight = compute_sample_weight('balanced', y_train)
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train)
    print(classes_weights)
    print(len(sampleWeight))


# In[ ]:





# In[ ]:





# # inital Model

# In[113]:


# XGBoost parameters
if device.type == "cuda":
    XGBparams = {
    'learning_rate': CFG.xgbLR, #0.05 #0.08, #0.01,#0.05,
    'max_depth': CFG.xgbDepth,  #5, #7, #6, low value avoid overfit
    'n_estimators':  CFG.xgbEstimate, # large number lead to overfit, small number lead to underfit
    'subsample': 0.8,
    'colsample_bytree': CFG.colsample_bytree, # 0.5, #0.8,
    'reg_alpha': 1,
    'reg_lambda': 5,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cuda',
    'verbosity': 0,
    }
   


else: # cpu base
     XGBparams = {
    'learning_rate': CFG.xgbLR, # 0.05 #0.08,#0.01, #0.05,
    'max_depth': CFG.xgbDepth, #7, #6, low value avoid overfit
    'n_estimators': CFG.xgbEstimate, #300, #200,  # large number lead to overfit, small number lead to underfit
    'subsample': 0.8,
    'colsample_bytree': CFG.colsample_bytree, #0.5, #0.8,
    'reg_alpha': 1,
    'reg_lambda': 5,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cpu',
    'verbosity': 0,
    }




# In[114]:


# xgb = XGBRFRegressor(**XGBparams ,  verbose=-1)
if CFG.crossValidate:
    xgb = XGBClassifier(**XGBparams ,  verbose=0, objective='multi:softmax', 
                    num_class=3, eval_metric=["merror","mlogloss"], early_stopping_rounds= CFG.xgbEarlyStop)
    tabNet = TabNetClassifier(
             n_d=CFG.n_d, n_a=CFG.n_a, n_steps=CFG.n_steps,
                        optimizer_params=dict(lr=CFG.learningRate),
                        optimizer_fn=torch.optim.Adam,
                        scheduler_params={"step_size":10, 
                                         "gamma":0.9},
                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                        mask_type= CFG.mask_type, #'entmax' # "sparsemax"
                        verbose= False,
                        
        )
    
    lgbm= LGBMClassifier(
            boosting_type="gbdt",
            device= "cuda" if torch.cuda.is_available() else "cpu",
            gpu_platform_id= 0,
            gpu_device_id= 0,
#             n_jobs= -1, # CPU
            random_state=1,
            learning_rate= 0.001,#0.04125, #learrning rate
            subsample=0.9,
            num_leaves =50,
            reg_alpha=0.05,
            reg_lambda=0.05,
            n_estimators=CFG.lgbmEstimators,
            max_depth= 5,
            verbose =-1,
            class_weight='balanced',
            verbosity= -1,
            )
    
    catBoost = CatBoostClassifier(
                thread_count=-1,
#                 task_type="GPU", not support gpu
#                 devices='0:1',
                random_state=42,
                # loss_function="CrossEntropy",
                loss_function='MultiClass', 
                verbose=False,
                learning_rate= 0.001,#0.04125,#0.04125, #learrning rate
                n_estimators=200,
                max_depth= 5,
               )

        
else:
    xgb = XGBClassifier(**XGBparams ,  verbose=0, objective='multi:softmax', 
                    num_class=3, sample_weight=sampleWeight, eval_metric=["merror","mlogloss"],
                        early_stopping_rounds= CFG.xgbEarlyStop)

    tabNet = TabNetClassifier(
             n_d=CFG.n_d, n_a=CFG.n_a, n_steps=CFG.n_steps,
                        optimizer_params=dict(lr=CFG.learningRate),
                        optimizer_fn=torch.optim.Adam,
                        scheduler_params={"step_size":10, 
                                         "gamma":0.9},
                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                        mask_type= CFG.mask_type, #'entmax' # "sparsemax"
                        
        )
    lgbm= LGBMClassifier(
            boosting_type="gbdt",
            device= "cuda" if torch.cuda.is_available() else "cpu",
            gpu_platform_id= 0,
            gpu_device_id= 0,
#             n_jobs= -1, # CPU
            random_state=1,
            learning_rate= 0.001,#0.04125, #learrning rate
            subsample=0.9,
            num_leaves =50,
            reg_alpha=0.05,
            reg_lambda=0.05,
            n_estimators=CFG.lgbmEstimators,
            max_depth= 5,
            class_weight='balanced',
            verbosity= -1)
    catBoost = CatBoostClassifier(
                thread_count=-1,
#                 task_type="GPU", not support gpu
#                 devices='0:1',
                random_state=42,
                # loss_function="CrossEntropy",
                loss_function='MultiClass', 
                verbose=False,
                learning_rate= 0.001,#0.04125,#0.04125, #learrning rate
                n_estimators=200,
                max_depth= 5,
               )


# In[115]:


xgb.get_params()


# In[116]:


lgbm


# In[117]:


tabNet


# In[118]:


catBoost.get_params()


# In[119]:


def plotConfustionMatrix(actualVal, predictVal, modelName, trainORVal="Training"):
    CM = confusion_matrix(actualVal, predictVal)
    classifyReport =  classification_report(actualVal, predictVal)
    classifyReportDF =  classification_report(actualVal, predictVal, output_dict=True)
    print(f"\n\rClassification Report For {modelName} {trainORVal}:\n\r", classifyReport)
    cmd =ConfusionMatrixDisplay(CM)
    cmd.plot()
    plt.title(f"Confusion Matrix for {modelName} {trainORVal}")
    plt.show()
    return classifyReportDF


# In[120]:


def plotKappa(trainKappa, valKappa, modelName):
    kFoldList = [item for item in range(1, len(trainKappa) + 1)]
    plt.figure(figsize=(6, 4))
    plt.plot(kFoldList, trainKappa, label='Training  Kappa')
    plt.plot(kFoldList, valKappa, label='Validation Kappa')
    plt.title(f'{modelName} Kappa (Training/Validation)')
    plt.xlabel('K-Fold')
    plt.ylabel('Kappa')
    plt.legend()
    plt.show()


# In[121]:


modelHistory= {} # record all trained Model


# In[122]:


# modelList =[("XGBoost", xgb ), ("LightBGM", lgbm), ("TabNet", tabNet), ("CatBoost", catBoost) ]
# modelList


# In[123]:


# for modelName, model  in modelList:
#     print("Name : ", modelName)
#     print("Model : ",model)


# # training Model
# ## Predict Train/Validation for Evalution Training performance

# In[124]:


if CFG.USE_ENSEMBLE:
    modelWeight = [0.25, 0.3, 0.25, 0.2]        


# In[125]:


def ensembleTrainValFunc():
    """
    for Ensemble learning with multiple model 
    """
    #history result collection
    xgbtrainKappaHist =[]
    lgbmtrainKappaHist =[]
    tabnettrainKappaHist =[]
    catBoosttrainKappaHist =[]
    
    xgbvalKappaHist = []
    lgbmvalKappaHist = []
    tabnetvalKappaHist = []
    catBoostvalKappaHist = []
    
    xgbtrainClassReportHist =[]
    lgbmtrainClassReportHist =[]
    tabnettrainClassReportHist =[]
    catBoosttrainClassReportHist =[]
    
    xgbvalClassReportHist =[]
    lgbmvalClassReportHist =[]
    tabnetvalClassReportHist =[]
    catBoostvalClassReportHist =[]
    
    if CFG.crossValidate:
        for i, (trainIdx, valIdx) in tqdm(enumerate(straKFold.split(xFeature, yLabel))):
            print(f"K-Fold: {i} ") 
            if CFG.normalize:
                X_train = xFeature[trainIdx, :] # filter/select train idx for X_Train
                X_test = xFeature[valIdx, :] # filter/select validation idx for X_Train
                y_train = yLabel[trainIdx]
                y_test  = yLabel[valIdx]
            else:
                X_train = xFeature.iloc[trainIdx, :] # filter/select train idx for X_Train
                X_test = xFeature.iloc[valIdx, :] # filter/select validation idx for X_Train
                y_train = yLabel.iloc[trainIdx]
                y_test  = yLabel.iloc[valIdx]
            
            # train all model
            print("Start Training XGBoost")
            evalSet = [(X_train, y_train), (X_test, y_test)]
            xgb.fit(X_train, y_train, eval_set= evalSet)
            print("Start Training LightGBM")
            lgbm.fit(X_train, y_train, eval_set= evalSet)
            print("Start Training TabNet")
            tabNet.fit(X_train.values, 
                        y_train,
                        patience= CFG.patience, max_epochs=CFG.maxEpochs,
                        eval_name=["Train", "Val"],
                        eval_set=[(X_train.values, y_train), (X_test.values, y_test)],
                        # eval_metric=['accuracy', "balanced_accuracy"],
                        eval_metric = ["logloss"],
                        weights=1,
                        drop_last=False)
            print("Start Training CatBoost")
            catBoost.fit(X_train, y_train, eval_set= evalSet, verbose=False)
            
            
            # predict train data 
            xgbYTrainPredict = xgb.predict(X_train)
            lgbmYTrainPredict = lgbm.predict(X_train)
            tabNetYTrainPredict = tabNet.predict(X_train.values)
            catBoostYTrainPredict = catBoost.predict(X_train)
            
            # predict validation data
            xgbYValPredict = xgb.predict(X_test)
            lgbmYValPredict = lgbm.predict(X_test)
            tabNetYValPredict = tabNet.predict(X_test.values)
            catBoostYValPredict = catBoost.predict(X_test)

            # plot XGBoost logloss
            results= xgb.evals_result()
            epochs = len(results['validation_0']['merror'])
            x_axis = range(0, epochs)
            # plot log loss
            fig, ax = plt.subplots()
            ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
            ax.plot(x_axis, results['validation_1']['mlogloss'], label='Val')
            ax.legend()
            plt.ylabel('Log Loss')
            plt.title('XGBoost Log Loss')
            plt.show()

            # plot Tab
            tabList = [item for item in range(1, len(tabNet.history["Train_logloss"]) + 1)]
            plt.figure(figsize=(6, 4))
            plt.plot(tabList, tabNet.history["Train_logloss"], label="Train")
            plt.plot(tabList, tabNet.history["Val_logloss"], label="Val")
            plt.title("TabNet Log Loss")
            plt.legend()
            plt.show()

            # plot catBoost
            catResult = catBoost.get_evals_result()
            plt.figure(figsize=(6, 4))
            plt.plot(catResult["validation_0"]["MultiClass"], label="Train")
            plt.plot(catResult["validation_1"]["MultiClass"], label="Val")
            plt.title("CatBoost Log Loss")
            plt.legend()
            plt.show()
  

            # plot LightGBM
            plt.figure(figsize=(6, 4))
            plt.plot(lgbm.evals_result_["training"]["multi_logloss"], label="Train")
            plt.plot(lgbm.evals_result_["valid_1"]["multi_logloss"], label="Val")
            plt.title("LightGBM Log Loss")
            plt.legend()
            plt.show()
            
            # Training Classification Report 
            trainClassifyReport = plotConfustionMatrix(y_train, xgbYTrainPredict, 
                                                       "XGBoost", trainORVal="Training")
            xgbtrainClassReportHist.append(trainClassifyReport)
            trainClassifyReport = plotConfustionMatrix(y_train, lgbmYTrainPredict, 
                                                       "LightGBM", trainORVal="Training")
            lgbmtrainClassReportHist.append(trainClassifyReport)
            trainClassifyReport = plotConfustionMatrix(y_train, tabNetYTrainPredict, 
                                                       "TabNet", trainORVal="Training")
            tabnettrainClassReportHist.append(trainClassifyReport)
            trainClassifyReport = plotConfustionMatrix(y_train, catBoostYTrainPredict, 
                                                       "CatBoost", trainORVal="Training")
            catBoosttrainClassReportHist.append(trainClassifyReport)
            
            
            # Validation Classification Report
            valClassifyReport = plotConfustionMatrix(y_test, xgbYValPredict, 
                                                       "XGBoost", trainORVal="Validation")
            xgbvalClassReportHist.append(valClassifyReport)
            valClassifyReport = plotConfustionMatrix(y_test, lgbmYValPredict, 
                                                       "LightGBM", trainORVal="Validation")
            lgbmvalClassReportHist.append(valClassifyReport)
        
            valClassifyReport = plotConfustionMatrix(y_test, tabNetYValPredict, 
                                                       "TabNet", trainORVal="Validation")
            tabnetvalClassReportHist.append(valClassifyReport)
            valClassifyReport = plotConfustionMatrix(y_test, catBoostYValPredict, 
                                                       "CatBoost", trainORVal="Validation")
            catBoostvalClassReportHist.append(valClassifyReport)

            # Train Kappa Score
            trainKappa = cohen_kappa_score(y_train, xgbYTrainPredict)
            xgbtrainKappaHist.append(trainKappa)            
            print(f"XGBoost Training cohen Kappa score: {trainKappa}")
            trainKappa = cohen_kappa_score(y_train, lgbmYTrainPredict)
            lgbmtrainKappaHist.append(trainKappa)            
            print(f"LightGBM Training cohen Kappa score: {trainKappa}")
            trainKappa = cohen_kappa_score(y_train, tabNetYTrainPredict)
            tabnettrainKappaHist.append(trainKappa)            
            print(f"TabNet Training cohen Kappa score: {trainKappa}")
            trainKappa = cohen_kappa_score(y_train, catBoostYTrainPredict)
            catBoosttrainKappaHist.append(trainKappa)            
            print(f"CatBoost Training cohen Kappa score: {trainKappa}")

            # Validation Kappa Score
            valKappa = cohen_kappa_score(y_test, xgbYValPredict)
            xgbvalKappaHist.append(valKappa)
            print(f"XGBoost Val cohen Kappa score: {valKappa}")
            valKappa = cohen_kappa_score(y_test, lgbmYValPredict)
            lgbmvalKappaHist.append(valKappa)
            print(f"LightGBM Val cohen Kappa score: {valKappa}")
            valKappa = cohen_kappa_score(y_test, tabNetYValPredict)
            tabnetvalKappaHist.append(valKappa)
            print(f"TabNet Val cohen Kappa score: {valKappa}")
            valKappa = cohen_kappa_score(y_test, catBoostYValPredict)
            catBoostvalKappaHist.append(valKappa)
            print(f"CatBoost Val cohen Kappa score: {valKappa}")
            
        
            
        #store is Model History
        modelHistory["XGBoost"] = {
            "k_Fold": CFG.nFold,
            "train_classify_report" : xgbtrainClassReportHist,
            "val_calssify_report": xgbvalClassReportHist,
            "train_kappa_score" : xgbtrainKappaHist,
            "val_kappa_score": xgbvalKappaHist
        }
        modelHistory["LightGBM"] = {
            "k_Fold": CFG.nFold,
            "train_classify_report" : lgbmtrainClassReportHist,
            "val_calssify_report": lgbmvalClassReportHist,
            "train_kappa_score" : lgbmtrainKappaHist,
            "val_kappa_score": lgbmvalKappaHist
        }
        modelHistory["TabNet"] = {
            "k_Fold": CFG.nFold,
            "train_classify_report" : tabnettrainClassReportHist,
            "val_calssify_report": tabnetvalClassReportHist,
            "train_kappa_score" : tabnettrainKappaHist,
            "val_kappa_score": tabnetvalKappaHist
        }
        modelHistory["CatBoost"] = {
            "k_Fold": CFG.nFold,
            "train_classify_report" : catBoosttrainClassReportHist,
            "val_calssify_report": catBoostvalClassReportHist,
            "train_kappa_score" : catBoosttrainKappaHist,
            "val_kappa_score": catBoostvalKappaHist
        }
            
            
#             
    else:
            # Train model
            print("Start Training XGBoost")
            evalSet = [(X_train, y_train), (X_test, y_test)]
            xgb.fit(X_train, y_train, eval_set= evalSet)
            print("Start Training LightGBM")
            lgbm.fit(X_train, y_train, eval_set= evalSet)
            print("Start Training TabNet")
            tabNet.fit(X_train.values, 
                        y_train,
                        patience= CFG.patience, max_epochs=CFG.maxEpochs,
                        eval_set=[(X_train.values, y_train), (X_test.values, y_test)],
                        # eval_metric=['accuracy', "balanced_accuracy"],
                        eval_metric = ["logloss"],
                        weights=1,
                        drop_last=False)
            print("Start Training CatBoost")
            catBoost.fit(X_train, y_train, eval_set= evalSet, verbose=False)
            
            
            # predict train data 
            xgbYTrainPredict = xgb.predict(X_train)
            lgbmYTrainPredict = lgbm.predict(X_train)
            tabNetYTrainPredict = tabNet.predict(X_train.values)
            catBoostYTrainPredict = catBoost.predict(X_train)
            
            # predict validation data
            xgbYValPredict = xgb.predict(X_test)
            lgbmYValPredict = lgbm.predict(X_test)
            tabNetYValPredict = tabNet.predict(X_test.values)
            catBoostYValPredict = catBoost.predict(X_test)

            # plot XGBoost logloss
            results= xgb.evals_result()
            epochs = len(results['validation_0']['merror'])
            x_axis = range(0, epochs)
            # plot log loss
            fig, ax = plt.subplots()
            ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
            ax.plot(x_axis, results['validation_1']['mlogloss'], label='Val')
            ax.legend()
            plt.ylabel('Log Loss')
            plt.title('XGBoost Log Loss')
            plt.show()

            # Training Classification Report 
            trainClassifyReport = plotConfustionMatrix(y_train, xgbYTrainPredict, 
                                                       "XGBoost", trainORVal="Training")
            xgbtrainClassReportHist.append(trainClassifyReport)
            trainClassifyReport = plotConfustionMatrix(y_train, lgbmYTrainPredict, 
                                                       "LightGBM", trainORVal="Training")
            lgbmtrainClassReportHist.append(trainClassifyReport)
            trainClassifyReport = plotConfustionMatrix(y_train, tabNetYTrainPredict, 
                                                       "TabNet", trainORVal="Training")
            tabnettrainClassReportHist.append(trainClassifyReport)
            trainClassifyReport = plotConfustionMatrix(y_train, catBoostYTrainPredict, 
                                                       "CatBoost", trainORVal="Training")
            catBoosttrainClassReportHist.append(trainClassifyReport)
            
            
            # Validation Classification Report
            valClassifyReport = plotConfustionMatrix(y_test, xgbYValPredict, 
                                                       "XGBoost", trainORVal="Validation")
            xgbvalClassReportHist.append(valClassifyReport)
            valClassifyReport = plotConfustionMatrix(y_test, lgbmYValPredict, 
                                                       "LightGBM", trainORVal="Validation")
            lgbmvalClassReportHist.append(valClassifyReport)
        
            valClassifyReport = plotConfustionMatrix(y_test, tabNetYValPredict, 
                                                       "TabNet", trainORVal="Validation")
            tabnetvalClassReportHist.append(valClassifyReport)
            valClassifyReport = plotConfustionMatrix(y_test, catBoostYValPredict, 
                                                       "CatBoost", trainORVal="Validation")
            catBoostvalClassReportHist.append(valClassifyReport)

            # Train Kappa Score
            trainKappa = cohen_kappa_score(y_train, xgbYTrainPredict)
            xgbtrainKappaHist.append(trainKappa)            
            print(f"XGBoost Training cohen Kappa score: {trainKappa}")
            trainKappa = cohen_kappa_score(y_train, lgbmYTrainPredict)
            lgbmtrainKappaHist.append(trainKappa)            
            print(f"LightGBM Training cohen Kappa score: {trainKappa}")
            trainKappa = cohen_kappa_score(y_train, tabNetYTrainPredict)
            tabnettrainKappaHist.append(trainKappa)            
            print(f"TabNet Training cohen Kappa score: {trainKappa}")
            trainKappa = cohen_kappa_score(y_train, catBoostYTrainPredict)
            catBoosttrainKappaHist.append(trainKappa)            
            print(f"CatBoost Training cohen Kappa score: {trainKappa}")

            # Validation Kappa Score
            valKappa = cohen_kappa_score(y_test, xgbYValPredict)
            xgbvalKappaHist.append(valKappa)
            print(f"XGBoost Val cohen Kappa score: {valKappa}")
            valKappa = cohen_kappa_score(y_test, lgbmYValPredict)
            lgbmvalKappaHist.append(valKappa)
            print(f"LightGBM Val cohen Kappa score: {valKappa}")
            valKappa = cohen_kappa_score(y_test, tabNetYValPredict)
            tabnetvalKappaHist.append(valKappa)
            print(f"TabNet Val cohen Kappa score: {valKappa}")
            valKappa = cohen_kappa_score(y_test, catBoostYValPredict)
            catBoostvalKappaHist.append(valKappa)
            print(f"CatBoost Val cohen Kappa score: {valKappa}")
            #store is Model History
            modelHistory["XGBoost"] = {
                "k_Fold": CFG.nFold,
                "train_classify_report" : xgbtrainClassReportHist,
                "val_calssify_report": xgbvalClassReportHist,
                "train_kappa_score" : xgbtrainKappaHist,
                "val_kappa_score": xgbvalKappaHist
            }
            modelHistory["LightGBM"] = {
                "k_Fold": CFG.nFold,
                "train_classify_report" : lgbmtrainClassReportHist,
                "val_calssify_report": lgbmvalClassReportHist,
                "train_kappa_score" : lgbmtrainKappaHist,
                "val_kappa_score": lgbmvalKappaHist
            }
            modelHistory["TabNet"] = {
                "k_Fold": CFG.nFold,
                "train_classify_report" : tabnettrainClassReportHist,
                "val_calssify_report": tabnetvalClassReportHist,
                "train_kappa_score" : tabnettrainKappaHist,
                "val_kappa_score": tabnetvalKappaHist
            }
            modelHistory["CatBoost"] = {
                "k_Fold": CFG.nFold,
                "train_classify_report" : catBoosttrainClassReportHist,
                "val_calssify_report": catBoostvalClassReportHist,
                "train_kappa_score" : catBoosttrainKappaHist,
                "val_kappa_score": catBoostvalKappaHist
            }


# In[126]:


# train single Model 
def trainValFunc(model , modelName):
    trainKappaHist =[]
    valKappaHist = []
    trainClassReportHist =[]
    valClassReportHist =[]
    
    if CFG.crossValidate:
        for i, (trainIdx, valIdx) in tqdm(enumerate(straKFold.split(xFeature, yLabel))):
            print(f"K-Fold: {i} ")
#             print(f"Train Idx : {trainIdx}")
#             print(f"Validate Idx: {valIdx}")
            # Set extract X_Train , X_test , y_train, y_test new split dataset 
            if CFG.normalize:
                X_train = xFeature[trainIdx, :] # filter/select train idx for X_Train
                X_test = xFeature[valIdx, :] # filter/select validation idx for X_Train
                y_train = yLabel[trainIdx]
                y_test  = yLabel[valIdx]
            else:
                X_train = xFeature.iloc[trainIdx, :] # filter/select train idx for X_Train
                X_test = xFeature.iloc[valIdx, :] # filter/select validation idx for X_Train
                y_train = yLabel.iloc[trainIdx]
                y_test  = yLabel.iloc[valIdx]
            # train model
            evalSet = [(X_train, y_train), (X_test, y_test)]
            model.fit(X_train, y_train, eval_set= evalSet, 
                      eval_metric=["merror","mlogloss"], early_stopping_rounds= CFG.xgbEarlyStop,  verbose=False)

            results= model.evals_result()
            epochs = len(results['validation_0']['merror'])
            x_axis = range(0, epochs)
            # plot log loss
            fig, ax = plt.subplots()
            ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
            ax.plot(x_axis, results['validation_1']['mlogloss'], label='Val')
            ax.legend()
            plt.ylabel('Log Loss')
            plt.title('XGBoost Log Loss')
            plt.show()
            # predict train dataset 
            yTrainPredict = model.predict(X_train)
#             print(yTrainPredict)
            yValPredict = model.predict(X_test)
#             print(yValPredict)
            trainClassifyReport = plotConfustionMatrix(y_train, yTrainPredict, 
                                                       modelName, trainORVal="Training")
            trainClassReportHist.append(trainClassifyReport)
            
            valClassifyReport = plotConfustionMatrix(y_test, yValPredict, 
                                                       modelName, trainORVal="Validation")
            valClassReportHist.append(valClassifyReport)
#             print(trainClassifyReport)
#             print(valClassifyReport)
            trainKappa = cohen_kappa_score(y_train, yTrainPredict)
            trainKappaHist.append(trainKappa)
            print(f"Training cohen Kappa score: {trainKappa}")
            valKappa = cohen_kappa_score(y_test, yValPredict)
            valKappaHist.append(valKappa)
            print(f"Val cohen Kappa score: {valKappa}")
            
        
            
        #store is Model History
        modelHistory[modelName] = {
            "k_Fold": CFG.nFold,
            "train_classify_report" : trainClassReportHist,
            "val_calssify_report": valClassReportHist,
            "train_kappa_score" : trainKappaHist,
            "val_kappa_score": valKappaHist
        }
            
            
#             
    else:
        # Train model
        evalSet = [(X_train, y_train), (X_test, y_test)]
        model.fit(X_train, y_train, eval_set= evalSet,  
                  early_stopping_rounds= CFG.xgbEarlyStop,  eval_metric=["merror","mlogloss"], verbose=False)
        # Train dateset prediction 
        yTrainPredict = model.predict(X_train)
#         print(yTrainPredict)
        yValPredict = model.predict(X_test)
#         print(yValPredict)
        trainClassifyReport = plotConfustionMatrix(y_train, yTrainPredict, 
                                                       modelName, trainORVal="Training")
        trainClassReportHist.append(trainClassifyReport)
        valClassifyReport = plotConfustionMatrix(y_test, yValPredict, 
                                                       modelName, trainORVal="Validation")
        valClassReportHist.append(valClassifyReport)
#         print(trainClassifyReport)
#         print(valClassifyReport)
        trainKappa = cohen_kappa_score(y_train, yTrainPredict)
        trainKappaHist.append(trainKappa)
        print(f"Training cohen Kappa score: {trainKappa}")
        valKappa = cohen_kappa_score(y_test, yValPredict)
        print(f"Val cohen Kappa score: {valKappa}")
        valKappaHist.append(valKappa)
        
        #store is Model History
        modelHistory[modelName] = {
            "train_classify_report" : trainClassReportHist,
            "val_calssify_report": valClassReportHist,
            "train_kappa_score" : trainKappaHist,
            "val_kappa_score": valKappaHist
        }


# In[127]:


get_ipython().run_cell_magic('time', '', 'if CFG.USE_ENSEMBLE:\n    ensembleTrainValFunc()\nelse:\n    trainValFunc(xgb, "XGBoost")\n')


# In[128]:


# tabNet.history["Val_logloss"]


# In[129]:


# if CFG.USE_ENSEMBLE:
#     # ensembleTrainValFunc()
#     for model, val in modelHistory.items():
#         print(f"Model: {model}\n\r{val}")
# else:
#     modelHistory["XGBoost"]


# In[130]:


# modelHistory["XGBoost"]["val_kappa_score"]


# In[131]:


for model in modelHistory.keys():
    trainKappa = np.mean(modelHistory[model]["train_kappa_score"])
    valKappa = np.mean(modelHistory[model]["val_kappa_score"])                     
    print(f"Average Training Kappa Score for {model} : {trainKappa}")
    print(f"Average Val Kappa Score for {model} :  {valKappa}")
    


# In[132]:


# print("Average Training Kappa Score for XGBoost :", np.mean(modelHistory["XGBoost"]["train_kappa_score"]))
# print("Average Val Kappa Score for XGBoost :", np.mean(modelHistory["XGBoost"]["val_kappa_score"]))
# print("Average Training Kappa Score for XGBoost :", np.mean(modelHistory["XGBoost"]["train_kappa_score"]))
# print("Average Val Kappa Score for XGBoost :", np.mean(modelHistory["XGBoost"]["val_kappa_score"]))


# In[133]:


for model in modelHistory.keys():
    plotKappa(modelHistory[model]["train_kappa_score"], 
          modelHistory[model]["val_kappa_score"], model)


# In[134]:


# plotKappa(modelHistory["XGBoost"]["train_kappa_score"], 
#           modelHistory["XGBoost"]["val_kappa_score"],
#          "XGBoost")


# In[135]:


# lgbm.evals_result_.keys()


# In[136]:


# plt.figure(figsize=(6, 4))
# plt.plot(lgbm.evals_result_["training"]["multi_logloss"], label="Train")
# plt.plot(lgbm.evals_result_["valid_1"]["multi_logloss"], label="Val")
# plt.title("LightGBM Log Loss")
# plt.legend()
# plt.show()


# ## Ensemble Learning Algriothm, use multi-model for improve 

# ## Explainable AI 
# #### Study and Understand How XGBoost trained model make decision

# In[137]:


feature_impXGB = pd.Series(xgb.feature_importances_, index=tempCol2).sort_values(ascending=False)
xgbTop20 = feature_impXGB[:30]
plt.figure(figsize=(10,6))
sns.barplot(x=xgbTop20, y=xgbTop20.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.title("Top 30 importenace feature for XGBoost");
plt.show()
# xgboost.plot_importance(xgb);


# In[138]:


plt.figure(figsize=(16, 16))
xgboost.plot_tree(xgb,  num_trees=0, rankdir='LR')
plt.savefig("xgb_out.png",  dpi=600)
plt.show();


# # Model Prediction for Submission

# In[139]:


# 
def roundoff(arr, thresholds=[0.5, 1.5, 2.5]):
    return np.where(arr < thresholds[0], 0, 
                np.where(arr < thresholds[1], 1, 
                    np.where(arr < thresholds[2], 2, 3)))


# In[140]:


if CFG.USE_ENSEMBLE:
    xgbYValPredict = xgb.predict(combinedTestDF)
    lgbmYValPredict = lgbm.predict(combinedTestDF)
    tabNetYValPredict = tabNet.predict(combinedTestDF.values)
    catBoostYValPredict = catBoost.predict(combinedTestDF)
    temp1 = np.add(modelWeight[0] * xgbYValPredict ,modelWeight[1] * lgbmYValPredict)
    temp2 = np.add(temp1, modelWeight[2] * tabNetYValPredict)
    submitPredict = np.add(temp2, (modelWeight[3] * np.squeeze(catBoostYValPredict)))
    submitPredict = roundoff(submitPredict)
else: 
     submitPredict = xgb.predict(combinedTestDF)
submitPredict


# In[141]:


roundedPredict = submitPredict # roundoff(submitPredict)
roundedPredict


# In[142]:


submit["sii"] = roundedPredict


# In[143]:


submit


# In[144]:


submit.to_csv('submission.csv', index=False)


# In[145]:


sub = pd.read_csv("submission.csv")
sub


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




