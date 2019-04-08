# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:13:27 2019

@author: Ricardo
"""
import pandas as pd
import pandas.core.algorithms as algos
from pandas import Series
from pandas import DataFrame

import numpy as np

from copy import deepcopy

from collections import defaultdict

from sklearn import metrics, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

from collections import OrderedDict

from statsmodels.stats.outliers_influence import variance_inflation_factor

from functools import reduce

import matplotlib.pyplot as plt

import scipy.stats.stats as stats

import re
import traceback

import warnings
warnings.filterwarnings("ignore")


# Binning Functions
def calculate_vif(features):
    vif = pd.DataFrame()
    vif["Features"] = features.columns
    vif["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]    
    return(vif)

def mono_bin(Y, X, n):
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df1, target, n):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i],n)
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)

# varSelection function: prepare the data and test multiple models for best fit
# args
#   df: pandas dataframe containing the complete dataset (features + target).
#       The target variable must be an int of 1 or 0 and the column must be
#       named as 'target' as well
def varSelection(df):
    print('\n\n\n################################################################')
    print('Starting engine...\n')
    print('\n\n\n################################################################')
    print('Descriptive Stats\n')
    print(df.target.value_counts())
    print(df.target.value_counts()/len(df))
    print(df.describe())
    
#    print('\nCorrelation graph\n')
#    %matplotlib inline
#    corr = df.corr()
#    print(sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns))
    print('\n\n\n################################################################')
    print('Data treatment\n')    
    # Replacing null of numeric variables by the mean value of the column
    for x in df.columns:
        print('Column {}: {}\n\tNull values: {}\n'.format(x,df[x].dtype,df[x].isnull().any()))
        if df[x].dtype in ('int32','int64','float32','float64') and df[x].isnull().any() == True:
            print('Replacing null values by the mean value of the whole column')
            df[x].fillna(df[x].mean(),inplace=True)
    
    # Converting categorical values to numeric using label encoder
    d = defaultdict(preprocessing.LabelEncoder)
    # Encoding the categorical variable
    print('Converting categorical values to numeric using label encoder')
    if df.select_dtypes(include=['object']).size > 0:
        fit = df.select_dtypes(include=['object']).fillna('NA').apply(lambda x: d[x.name].fit_transform(x))
        #Convert the categorical columns based on encoding
        for i in list(d.keys()):
            df[i] = d[i].transform(df[i].fillna('NA'))
    print('\n\n\n################################################################')
    print('Spliting the dataframe into features and labels\n')
    features = df[df.columns.difference(['target'])]
    labels = df['target']
    features = features.fillna(0)
    print('\n\n\n################################################################')
    # WOE - Weight of Evidence and IV - Information Value
    print('\nVariable Selection (WOE and IV)')
    
    final_iv, IV = data_vars(df[df.columns.difference(['target'])],df.target,max_bin)
    IV = IV.rename(columns={'VAR_NAME':'index'})
    print('----\n|IV|\n----\n{}'.format(IV.sort_values(['IV'],ascending=0)))
    print('-----\n|WOE|\n-----\n')
    transform_vars_list = df.columns.difference(['target'])
    transform_prefix = 'new_'
    for var in transform_vars_list:
        small_df = final_iv[final_iv['VAR_NAME'] == var]
        transform_dict = dict(zip(small_df.MAX_VALUE,small_df.WOE))
        replace_cmd = ''
        replace_cmd1 = ''
        for i in sorted(transform_dict.items()):
            replace_cmd = replace_cmd + str(i[1]) + str(' if x <= ') + str(i[0]) + ' else '
            replace_cmd1 = replace_cmd1 + str(i[1]) + str(' if x == "') + str(i[0]) + '" else '
        replace_cmd = replace_cmd + '0'
        replace_cmd1 = replace_cmd1 + '0'
        if replace_cmd != '0':
            try:
                df[transform_prefix + var] = df[var].apply(lambda x: eval(replace_cmd))
            except:
                df[transform_prefix + var] = df[var].apply(lambda x: eval(replace_cmd1))
    # Randon Forest
    clf = RandomForestClassifier(n_estimators=nEstimatorsRF)
    clf.fit(features,labels)   
    preds = clf.predict(features)    
    accuracy = accuracy_score(preds,labels)
    print('\n\n\nRandon Forest - n_estimators: {} - Accuracy: {}'.format(nEstimatorsRF,accuracy))
    VI = DataFrame(clf.feature_importances_, columns = ["RF"], index=features.columns)
    VI = VI.reset_index()
    print(VI.sort_values(['RF'],ascending=0))
    
    # Recursive Feature Elimination
    model = LogisticRegression(solver='lbfgs',max_iter=maxIterRFE)
    rfe = RFE(model, 20)
    fit = rfe.fit(features, labels)
    Selected = DataFrame(rfe.support_, columns = ["RFE"], index=features.columns)
    Selected = Selected.reset_index()
    print('\n\n\nRecursive Feature Elimination - max_iter: {}'.format(maxIterRFE))
    print(Selected[Selected['RFE'] == True])
    
    # ExtraTrees Classifier
    model = ExtraTreesClassifier(n_estimators=nEstimatorsETC)
    model.fit(features, labels)
    FI = DataFrame(model.feature_importances_, columns = ["Extratrees"], index=features.columns)
    FI = FI.reset_index()
    print('\n\n\nExtra Trees Classifier - n_estimators: {}'.format(nEstimatorsETC))
    print(FI.sort_values(['Extratrees'],ascending=0))
    
    # Chi Square
    model = SelectKBest(score_func=chi2, k=kChiSq)
    fit = model.fit(features.abs(), labels)
    pd.options.display.float_format = '{:.2f}'.format
    chi_sq = DataFrame(fit.scores_, columns = ["Chi_Square"], index=features.columns)
    chi_sq = chi_sq.reset_index()
    print('\n\n\nChi Square - k: {}'.format(kChiSq))  
    print(chi_sq.sort_values('Chi_Square',ascending=0))
    
    # L1
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=maxIterL1).fit(features, labels)
    model = SelectFromModel(lsvc,prefit=True)
    l1 = DataFrame(model.get_support(), columns = ["L1"], index=features.columns)
    l1 = l1.reset_index()
    print('\n\n\nL1 - max_iter: {}'.format(maxIterL1))  
    print(l1[l1['L1'] == True])
    
    # Combining
    dfs = [IV, VI, Selected, FI, chi_sq, l1]
    final_results = reduce(lambda left,right: pd.merge(left,right,on='index'), dfs)
    columns = ['IV', 'RF', 'Extratrees', 'Chi_Square']
    score_table = pd.DataFrame({},[])
    score_table['index'] = final_results['index'] 
    for i in columns:
        score_table[i] = final_results['index'].isin(list(final_results.nlargest(5,i)['index'])).astype(int)     
    score_table['RFE'] = final_results['RFE'].astype(int)
    score_table['L1'] = final_results['L1'].astype(int)
    score_table['final_score'] = score_table.sum(axis=1)
    print('\n\n\nScore Table')
    print(score_table.sort_values('final_score',ascending=0))
    
    # Multicollinearity
    features = features[list(score_table[score_table['final_score'] >= minVarScore]['index'])]
    vif = calculate_vif(features)
    if maxVarianceInflationFactor > 0:
        while vif['VIF'][vif['VIF'] > maxVarianceInflationFactor].any():
            remove = vif.sort_values('VIF',ascending=0)['Features'][:1]
            features.drop(remove,axis=1,inplace=True)
            vif = calculate_vif(features)
    print('\n\n\nMulticollinearity - minVarScore: {} - maxVarianceInflationFactor: {}'.format(minVarScore,maxVarianceInflationFactor))
    final_vars = list(vif['Features']) + ['target']
    df1 = df[final_vars].fillna(0)
    print(df1.describe())
    
#    bar_color = '#058caa'
#    num_color = '#ed8549'
#    final_iv,_ = data_vars(df1,df1['target'],max_bin)
#    final_iv = final_iv[(final_iv.VAR_NAME != 'target')]
#    grouped = final_iv.groupby(['VAR_NAME'])
#    for key, group in grouped:
#        ax = group.plot('MIN_VALUE','EVENT_RATE',kind='bar',color=bar_color,linewidth=1.0,edgecolor=['black'])
#        ax.set_title(str(key) + " vs " + str('target'))
#        ax.set_xlabel(key)
#        ax.set_ylabel(str('target') + " %")
#        rects = ax.patches
#        for rect in rects:
#            height = rect.get_height()
#            ax.text(rect.get_x()+rect.get_width()/2., 1.01*height, str(round(height*100,1)) + '%', 
#                    ha='center', va='bottom', color=num_color, fontweight='bold')

    return df1, vif, d

def plot_pandas_style(styler):
    from IPython.core.display import HTML
    html = '\n'.join([line.lstrip() for line in styler.render().split('\n')])
    return HTML(html)

def highlight_max(s,color='yellow'):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: {}'.format(color) if v else '' for v in is_max]

def decile_labels(agg1,label,color='skyblue'):
    agg_dummy = pd.DataFrame(OrderedDict((('TOTAL',0),('TARGET',0),('NONTARGET',0),('PCT_TAR',0),('CUM_TAR',0),('CUM_NONTAR',0),('DIST_TAR',0),('DIST_NONTAR',0),('SPREAD',0))),index=[0])
    agg1 = agg1.append(agg_dummy).sort_index()
    agg1.index.name = label
    agg1 = agg1.style.apply(highlight_max, color = 'yellow', subset=['SPREAD'])
    agg1.bar(subset=['TARGET'], color='{}'.format(color))
    agg1.bar(subset=['TOTAL'], color='{}'.format(color))
    agg1.bar(subset=['PCT_TAR'], color='{}'.format(color))
    return(agg1)

def deciling(data,decile_by,target,nontarget):
    inputs = list(decile_by)
    inputs.extend((target,nontarget))
    decile = data[inputs]
    grouped = decile.groupby(decile_by)
    agg1 = pd.DataFrame({},index=[])
    agg1['TOTAL'] = grouped.sum()[nontarget] + grouped.sum()[target]
    agg1['TARGET'] = grouped.sum()[target]
    agg1['NONTARGET'] = grouped.sum()[nontarget]
    agg1['PCT_TAR'] = grouped.mean()[target]*100
    agg1['CUM_TAR'] = grouped.sum()[target].cumsum()
    agg1['CUM_NONTAR'] = grouped.sum()[nontarget].cumsum()
    agg1['DIST_TAR'] = agg1['CUM_TAR']/agg1['TARGET'].sum()*100
    agg1['DIST_NONTAR'] = agg1['CUM_NONTAR']/agg1['NONTARGET'].sum()*100
    agg1['SPREAD'] = (agg1['DIST_TAR'] - agg1['DIST_NONTAR'])
    agg1 = decile_labels(agg1,'DECILE',color='skyblue')
    return(plot_pandas_style(agg1))
    
def scoring(features,clf,target):
    score = pd.DataFrame(clf.predict_proba(features)[:,1], columns = ['SCORE'])
    score['DECILE'] = pd.qcut(score['SCORE'].rank(method = 'first'),10,labels=range(10,0,-1))
    score['DECILE'] = score['DECILE'].astype(float)
    score['TARGET'] = target
    score['NONTARGET'] = 1 - target
    return(score)
    
def plots(agg1,target,type):

    plt.figure(1,figsize=(20, 5))

    plt.subplot(131)
    plt.plot(agg1['DECILE'],agg1['ACTUAL'],label='Actual')
    plt.plot(agg1['DECILE'],agg1['PRED'],label='Pred')
    plt.xticks(range(10,110,10))
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.title('Actual vs Predicted', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel(str(target) + " " + str(type) + " %",fontsize=15)

    plt.subplot(132)
    X = agg1['DECILE'].tolist()
    X.append(0)
    Y = agg1['DIST_TAR'].tolist()
    Y.append(0)
    plt.plot(sorted(X),sorted(Y))
    plt.plot([0, 100], [0, 100],'r--')
    plt.xticks(range(0,110,10))
    plt.yticks(range(0,110,10))
    plt.grid(True)
    plt.title('Gains Chart', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel(str(target) + str(" DISTRIBUTION") + " %",fontsize=15)
    plt.annotate(round(agg1[agg1['DECILE'] == 30].DIST_TAR.item(),2),xy=[30,30], 
            xytext=(25, agg1[agg1['DECILE'] == 30].DIST_TAR.item() + 5),fontsize = 13)
    plt.annotate(round(agg1[agg1['DECILE'] == 50].DIST_TAR.item(),2),xy=[50,50], 
            xytext=(45, agg1[agg1['DECILE'] == 50].DIST_TAR.item() + 5),fontsize = 13)

    plt.subplot(133)
    plt.plot(agg1['DECILE'],agg1['LIFT'])
    plt.xticks(range(10,110,10))
    plt.grid(True)
    plt.title('Lift Chart', fontsize=20)
    plt.xlabel("Population %",fontsize=15)
    plt.ylabel("Lift",fontsize=15)

    plt.tight_layout()



def gains(data,decile_by,target,score):
    inputs = list(decile_by)
    inputs.extend((target,score))
    decile = data[inputs]
    grouped = decile.groupby(decile_by)
    agg1 = pd.DataFrame({},index=[])
    agg1['ACTUAL'] = grouped.mean()[target]*100
    agg1['PRED'] = grouped.mean()[score]*100
    agg1['DIST_TAR'] = grouped.sum()[target].cumsum()/grouped.sum()[target].sum()*100
    agg1.index.name = 'DECILE'
    agg1 = agg1.reset_index()
    agg1['DECILE'] = agg1['DECILE']*10
    agg1['LIFT'] = agg1['DIST_TAR']/agg1['DECILE']
    plots(agg1,target,'Distribution')


def runModel(modelType,features_train,label_train,features_test,label_test):
    
    def runClf(): 
        clf.fit(features_train,label_train)
        
        pred_train = clf.predict(features_train)
        pred_test = clf.predict(features_test)
        
        
        accuracy_train = accuracy_score(pred_train,label_train)
        accuracy_test = accuracy_score(pred_test,label_test)
        
        fpr, tpr, _ = metrics.roc_curve(np.array(label_train), clf.predict_proba(features_train)[:,1])
        auc_train = metrics.auc(fpr,tpr)
        
        fpr, tpr, _ = metrics.roc_curve(np.array(label_test), clf.predict_proba(features_test)[:,1])
        auc_test = metrics.auc(fpr,tpr)
        
        print('accuracy_train: {}\naccuracy_test: {}\nauc_train: {}\nauc_test: {}'.format(accuracy_train,accuracy_test,auc_train,auc_test))
        print('\n\n\nAssertivity - Train\n')
        print(pd.crosstab(label_train,pd.Series(pred_train),rownames=['ACTUAL'],colnames=['PRED']))
        print('\n\n\nAssertivity - Test\n')
        print(pd.crosstab(label_test,pd.Series(pred_test),rownames=['ACTUAL'],colnames=['PRED']))
        
        scores_train = scoring(features_train,clf,label_train)
        scores_test = scoring(features_test,clf,label_test)
        lift_train = pd.concat([features_train,scores_train],axis=1)
        lift_test = pd.concat([features_test,scores_test],axis=1)
        
        return clf, scores_train, scores_test, lift_train, lift_test
    
    if modelType == 'RF':
        print('Random Forest Classifier - n_estimators: {}'.format(nEstimatorsRF))
        clf = RandomForestClassifier(n_estimators=nEstimatorsRF)
        return runClf()
    elif modelType == 'LR':
        print('Logistic Regression')
        clf = LogisticRegression(solver='lbfgs')   
        return runClf()
    elif modelType == 'NN':
        print('Neural Network')
        clf = MLPClassifier()
        return runClf()
    elif modelType == 'NB':
        print('Naive Bayes')
        clf = GaussianNB()
        return runClf()
    elif modelType == 'GB':
        print('Gradient Boost')    
        clf = GradientBoostingClassifier()
        return runClf()
    elif modelType == 'RFHP':
        print('Random Forest - Hyper Parameter Tuning')
        n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(3, 10, num = 1)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        
        rf = RandomForestClassifier()
        
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 2, verbose=2, random_state=42, n_jobs = -1)
        rf_random.fit(features_train, label_train)
        
        print(rf_random.best_params_)
        clf = RandomForestClassifier(**rf_random.best_params_)
        return runClf()
    elif modelType == 'GBHP':
        print('Gradient Boost - Hyper Parameter Tuning')
        n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(3, 10, num = 1)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        
        grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
        
        gb = GradientBoostingClassifier()
        
        gf_tune = GridSearchCV(estimator = gb, param_grid = grid, cv = 2, verbose=2, n_jobs = -1)
        gf_tune.fit(features_train, label_train)
        
        print(gf_tune.best_params_)
        clf = GradientBoostingClassifier(**gf_tune.best_params_)
        return runClf()


def printCharts(scores_train,scores_test,lift_train,lift_test):
    print(deciling(scores_train,['DECILE'],'TARGET','NONTARGET'))
    print(deciling(scores_test,['DECILE'],'TARGET','NONTARGET'))
    print(gains(lift_train,['DECILE'],'TARGET','SCORE'))
    print(gains(lift_test,['DECILE'],'TARGET','SCORE'))

def saveModel(file,dictonary,model):
    filename = '{}.model'.format(file)
    i = [dictonary,model]
    joblib.dump(i,filename)

def modelSelection(df1,vif):
    print('\n\n\n################################################################')
    print('Spliting the dataframe into train and test - test_size: {}'.format(testSize))
    train, test = train_test_split(df1, test_size = testSize)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    fTrain = train[list(vif['Features'])]
    lTrain = train['target']
    fTest = test[list(vif['Features'])]
    lTest = test['target']
    
    clfRF, scores_train, scores_test, lift_train, lift_test = runModel('RF',fTrain,lTrain,fTest,lTest) #RandomForest
    #printCharts(scores_train,scores_test,lift_train,lift_test)
    
    clfLR, scores_train, scores_test, lift_train, lift_test = runModel('LR',fTrain,lTrain,fTest,lTest) #LogisticRegression
    #printCharts(scores_train,scores_test,lift_train,lift_test)
    
    clfNN, scores_train, scores_test, lift_train, lift_test = runModel('NN',fTrain,lTrain,fTest,lTest) #NeuralNetwork
    #printCharts(scores_train,scores_test,lift_train,lift_test)
    
    clfNB, scores_train, scores_test, lift_train, lift_test = runModel('NB',fTrain,lTrain,fTest,lTest) #NaiveBayes
    #printCharts(scores_train,scores_test,lift_train,lift_test)
    
    clfGB, scores_train, scores_test, lift_train, lift_test = runModel('GB',fTrain,lTrain,fTest,lTest) #GradientBoost
    #printCharts(scores_train,scores_test,lift_train,lift_test)
    
    clfRFHP, scores_train, scores_test, lift_train, lift_test = runModel('RFHP',fTrain,lTrain,fTest,lTest) #RandomForestHyperParameter
    #printCharts(scores_train,scores_test,lift_train,lift_test)
    
    clfGBHP, scores_train, scores_test, lift_train, lift_test = runModel('GBHP',fTrain,lTrain,fTest,lTest) #GradientBoostHyperParameter
    #printCharts(scores_train,scores_test,lift_train,lift_test)
    
    return clfRF, clfLR, clfNN, clfNB, clfGB, clfRFHP, clfGBHP
    

max_bin = 20
force_bin = 3
nEstimatorsRF = 720
nEstimatorsETC = 720
maxIterRFE = 100000
maxIterL1 = 100000
kChiSq = 'all'
minVarScore = 2
maxVarianceInflationFactor = 10 # default=10, ignore=-1
testSize = 0.4

dfInput = pd.read_csv('path/to/csv/with/features/and/target')

dfModel, vif, dictonary = varSelection(dfInput)

clfRF, clfLR, clfNN, clfNB, clfGB, clfRFHP, clfGBHP = modelSelection(dfModel,vif)
    