
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     readonly/train.csv - the training set (all tickets issued 2004-2011)
#     readonly/test.csv - the test set (all tickets issued 2012-2016)
#     readonly/addresses.csv & readonly/latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `readonly/train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `readonly/test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#        
# ### Hints
# 
# * Make sure your code is working before submitting it to the autograder.
# 
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
# 
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question. 
# 
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
# 
# * Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.

# In[16]:

def blight_model():    
    import pandas as pd
    import numpy as np

    train = pd.read_csv('train.csv', encoding = "ISO-8859-1")
    test = pd.read_csv('test.csv')
    addresses = pd.read_csv('addresses.csv')
    latlons = pd.read_csv('latlons.csv')

    # def blight_model():

    #train.shape

    #test.shape

    #train.columns

    #for x in train.columns:
        #print(x, train[x].nunique(), train[x].isnull().sum()) ##as vrea sa-l fac un tabel

    #train.info()

    #train.head()

    #addresses.head()

    #latlons.head()

    #discount_amount -> discount_procentage
    #voi adauga lat si lon
    useful_columns = ['ticket_id', 'agency_name', 'inspector_name', 'city', 'state', 'violation_code',
                     'violation_description', 'disposition', 'judgment_amount', 'discount_amount', 
                      'compliance']
    useful_columns2 = ['ticket_id', 'agency_name', 'inspector_name', 'city', 'state', 'violation_code',
                     'violation_description', 'disposition', 'judgment_amount', 'discount_amount']

    train_copy1 = train[useful_columns].copy()
    test_copy1 = test[useful_columns2].copy()

    #train_copy1.head()

    #for x in train_copy1:
        #print(x, train_copy1[x].nunique(), train_copy1[x].isnull().sum())

    #train_copy1['city'].unique()

    train_copy1 = train_copy1.apply(lambda x: x.astype(str).str.upper())

    #for x in train_copy1.columns:
        #print(x, train_copy1[x].nunique())

    #train_copy1['city'].value_counts()

    train_copy2 = train_copy1.copy()

    mask = train_copy2.city.map(train_copy1.city.value_counts()) < 800

    train_copy2.city = train_copy1.city.mask(mask, 'OTHER')

    test_copy1 = test_copy1.apply(lambda x: x.astype(str).str.upper())

    #test_copy1.head()

    #cities_train = train_copy2.city.unique()

    #cities_test = test_copy1.city.unique()

    #for x in cities_train:
        #if (x != 'OTHER') and x not in cities_test:
           # print(x)

    #train_copy2.city.value_counts()


    train_copy2['city'] = train_copy2['city'].replace(['DET', 'DET.'], 'DETROIT' )

    test_copy1['city'] = test_copy1['city'].replace(['DET', 'DET.'], 'DETROIT' )

    cities_train = train_copy2.city.unique()
    cities_test = test_copy1.city.unique()
    #for x in cities_train:
       # if (x != 'OTHER') and x not in cities_test:
            #print(x)

    test_copy2 = test_copy1.copy()

    test_copy2['city'] = test_copy2['city'].replace([x for x in cities_test if x not in cities_train], 'OTHER' )

    #test_copy2.city.value_counts()

    #train_copy2.shape

    #test_copy2.shape

    ## adaug lat si lon
    ## encodere pe str uri
    ## scalare

    #for x in train_copy2.columns:
        #print(x, train_copy2[x].nunique())

    #train_copy2.head()

    #train_copy2.shape

    train_copy3 = train_copy2.copy()

    #train_copy3.head()

    addresses_latlons = pd.merge(latlons, addresses, on='address')

    #addresses.head()

    #latlons.head()

    #addresses_latlons.shape

    #addresses_latlons.head()

    #train_copy3 = pd.merge(addresses_latlons, train_copy3, on = "ticket_id")

    #train_copy3.head()

    #i=0
    #for x in train_copy3['ticket_id']:
        #if x in addresses_latlons['ticket_id']:
            #i += 1
    #print(i)

    #train_copy3.dtypes

    #addresses_latlons.dtypes

    train_copy3.ticket_id = train_copy3.ticket_id.astype('int64')
    train_copy3.judgment_amount = train_copy3.judgment_amount.astype('float64')
    train_copy3.discount_amount = train_copy3.discount_amount.astype('float64')


    #train_copy3.dtypes

    train_copy3 = pd.merge(train_copy3, addresses_latlons , on='ticket_id')

    #train_copy3.head()

    #train_copy3.shape

    train_copy3.drop('address', axis =1)

    #train_copy3['lat'].isnull().sum()

    #train_copy3['lon'].isnull().sum()

    train_copy4 = train_copy3.copy()

    train_copy4 = train_copy4.dropna(axis = 0)

    #train_copy4.head()

    #for x in train_copy4.columns:
        #print(x, train_copy4[x].isnull().sum())

    train_copy4.drop('address', axis='columns', inplace=True)

    #train_copy4.info()

    columns_object = ['agency_name', 'inspector_name', 'city', 'state', 'violation_code', 
                     'violation_description', 'disposition']

    #for x in columns_object:
        #print(x, train_copy4[x].nunique())

    #train_copy4.disposition.unique()

    ## one-hot -> agency_name
    ## label -> desposition
    ## drop -> inspector_name, state, violation_code, violation_description

    train_copy5 = train_copy4.copy()

    train_copy5.drop(['inspector_name', 'state', 
                       'violation_description'], axis='columns', inplace=True)

    #train_copy5.head()

    #train_copy5.dtypes

    #test_copy2.head()

    test_copy3 = test_copy2.drop(['inspector_name', 'state', 
                       'violation_description'], axis = 1)

    #test_copy3.head()

    #test_copy3.dtypes

    test_copy3.ticket_id = test_copy3.ticket_id.astype('int64')

    #test_copy3.dtypes

    test_copy3.judgment_amount = test_copy3.judgment_amount.astype('float64')
    test_copy3.discount_amount = test_copy3.discount_amount.astype('float64')

    #test_copy3.dtypes

    #train_copy5.dtypes

    #test_copy3.info()

    test_copy3 = pd.merge(test_copy3, addresses_latlons , on='ticket_id')

    #test_copy3.head()

    test_copy4 = test_copy3.drop(['address'], axis = 1)

    #test_copy4.info()

    test_copy4 = test_copy4.fillna(test_copy4.mean(), inplace = True)

    #test_copy4.info()


    from sklearn.preprocessing import LabelEncoder

    train_copy5 = train_copy5[ (train_copy5['compliance'] == '0.0') | (train_copy5['compliance'] == '1.0')]

    label_encoder = LabelEncoder()
    label_encoder.fit(train_copy5['disposition'].append(test_copy4['disposition'], ignore_index=True))
    train_copy5['disposition'] = label_encoder.transform(train_copy5['disposition'])
    test_copy4['disposition'] = label_encoder.transform(test_copy4['disposition'])

    label_encoder = LabelEncoder()
    label_encoder.fit(train_copy5['agency_name'].append(test_copy4['agency_name'], ignore_index=True))
    train_copy5['agency_name'] = label_encoder.transform(train_copy5['agency_name'])
    test_copy4['agency_name'] = label_encoder.transform(test_copy4['agency_name'])

    label_encoder = LabelEncoder()
    label_encoder.fit(train_copy5['city'].append(test_copy4['city'], ignore_index=True))
    train_copy5['city'] = label_encoder.transform(train_copy5['city'])
    test_copy4['city'] = label_encoder.transform(test_copy4['city'])

    label_encoder = LabelEncoder()
    label_encoder.fit(train_copy5['violation_code'].append(test_copy4['violation_code'], ignore_index=True))
    train_copy5['violation_code'] = label_encoder.transform(train_copy5['violation_code'])
    test_copy4['violation_code'] = label_encoder.transform(test_copy4['violation_code'])


    #train_copy5.info()

    train_copy6 = train_copy5.copy()

    #train_copy6.info()

    #train_copy6.columns

    X = train_copy6[['ticket_id', 'agency_name', 'city', 'violation_code', 'disposition',
           'judgment_amount', 'discount_amount', 'lat', 'lon']]

    train_copy6.compliance = train_copy6.compliance.astype('float64')

    y = train_copy6['compliance']

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import roc_auc_score
    import sklearn.metrics


    regr_rf = RandomForestRegressor()
    grid_values = {'n_estimators': [10, 100], 'max_depth': [None, 30]}
    grid_clf_auc = GridSearchCV(regr_rf, param_grid=grid_values, scoring='roc_auc')
    grid_clf_auc.fit(X_train, y_train)
    print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
    print('Grid best score (AUC): ', grid_clf_auc.best_score_)

    from sklearn import metrics

    y_pred = grid_clf_auc.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    auc_scr = metrics.auc(fpr, tpr)

    #print(fpr)
    #print(tpr)

    #print('AUC pt test este : ', auc_scr)
    
    return pd.DataFrame(grid_clf_auc.predict(test_copy4), test_copy4.ticket_id)

