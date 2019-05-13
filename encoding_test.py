import pandas as pd
import numpy as np

train = pd.read_csv("./input/titanic/train.csv")
test = pd.read_csv("./input/titanic/test.csv")

train["female"] = train["Sex"].apply(lambda x:1 if x=='female' else 0)
train.drop("Sex",axis=1,inplace=True)
test["female"] = test["Sex"].apply(lambda x:1 if x=='female' else 0)
test.drop("Sex",axis=1,inplace=True)

male_class1_age_mean = train[(train["female"]==0) & (train["Pclass"]==1)]["Age"].mean()
male_class2_age_mean = train[(train["female"]==0) & (train["Pclass"]==2)]["Age"].mean()
male_class3_age_mean = train[(train["female"]==0) & (train["Pclass"]==3)]["Age"].mean()
male_class1_age_std = train[(train["female"]==0) & (train["Pclass"]==1)]["Age"].std()
male_class2_age_std = train[(train["female"]==0) & (train["Pclass"]==2)]["Age"].std()
male_class3_age_std = train[(train["female"]==0) & (train["Pclass"]==3)]["Age"].std()

female_class1_age_mean = train[(train["female"]==1) & (train["Pclass"]==1)]["Age"].mean()
female_class2_age_mean = train[(train["female"]==1) & (train["Pclass"]==2)]["Age"].mean()
female_class3_age_mean = train[(train["female"]==1) & (train["Pclass"]==3)]["Age"].mean()
female_class1_age_std = train[(train["female"]==1) & (train["Pclass"]==1)]["Age"].std()
female_class2_age_std = train[(train["female"]==1) & (train["Pclass"]==2)]["Age"].std()
female_class3_age_std = train[(train["female"]==1) & (train["Pclass"]==3)]["Age"].std()

train["Age"] = train["Age"].fillna(-1)
test["Age"] = test["Age"].fillna(-1)

for i in range(train.shape[0]):
    if train.iloc[i]["Age"] == -1:
        if (train.iloc[i]['female']==0) & (train.iloc[i]['Pclass']==1):
            train.iat[i,4] = int(np.random.normal(male_class1_age_mean,male_class1_age_std))
        elif (train.iloc[i]['female']==0) & (train.iloc[i]['Pclass']==2):
            train.iat[i,4] = int(np.random.normal(male_class2_age_mean,male_class2_age_std))
        elif (train.iloc[i]['female']==0) & (train.iloc[i]['Pclass']==3):
            train.iat[i,4] = int(np.random.normal(male_class3_age_mean,male_class3_age_std))
        elif (train.iloc[i]['female']==1) & (train.iloc[i]['Pclass']==1):
            train.iat[i,4] = int(np.random.normal(female_class1_age_mean,female_class1_age_std))
        elif (train.iloc[i]['female']==1) & (train.iloc[i]['Pclass']==2):
            train.iat[i,4] = int(np.random.normal(female_class2_age_mean,female_class2_age_std))
        elif (train.iloc[i]['female']==1) & (train.iloc[i]['Pclass']==3):
            train.iat[i,4] =int(np.random.normal(female_class3_age_mean,female_class3_age_std))
    else:
        pass

for i in range(test.shape[0]):
    if test.iloc[i]["Age"] == -1:
        if (test.iloc[i]['female']==0) & (test.iloc[i]['Pclass']==1):
            test.iat[i,4] = np.random.normal(male_class1_age_mean,male_class1_age_std)
        elif (test.iloc[i]['female']==0) & (test.iloc[i]['Pclass']==2):
            test.iat[i,4] = np.random.normal(male_class2_age_mean,male_class2_age_std)
        elif (test.iloc[i]['female']==0) & (test.iloc[i]['Pclass']==3):
            test.iat[i,4] = np.random.normal(male_class3_age_mean,male_class3_age_std)
        elif (test.iloc[i]['female']==1) & (test.iloc[i]['Pclass']==1):
            test.iat[i,4] = np.random.normal(female_class1_age_mean,female_class1_age_std)
        elif (test.iloc[i]['female']==1) & (test.iloc[i]['Pclass']==2):
            test.iat[i,4] = np.random.normal(female_class2_age_mean,female_class2_age_std)
        elif (test.iloc[i]['female']==1) & (test.iloc[i]['Pclass']==3):
            test.iat[i,4] = np.random.normal(female_class3_age_mean,female_class3_age_std)
    else:
        pass

train["family"] = train["SibSp"] + train["Parch"]
train.drop(["SibSp","Parch","Name","Ticket","Cabin"],axis=1,inplace=True)
test["family"] = test["SibSp"] + test["Parch"]
test.drop(["SibSp","Parch","Name","Ticket","Cabin"],axis=1,inplace=True)
train["isAlone"] = train["family"].apply(lambda x:1 if x==0 else 0)
test["isAlone"] = test["family"].apply(lambda x:1 if x==0 else 0)

train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")
test["Fare"] = test["Fare"].fillna(np.mean(test["Fare"]))

#target_encording
te_embarked_map = {"S":np.mean(train[train["Embarked"]=="S"]["Survived"]),"C":np.mean(train[train["Embarked"]=="C"]["Survived"]),"Q":np.mean(train[train["Embarked"]=="Q"]["Survived"])}
train["te_embarked"] = train["Embarked"].apply(lambda x: te_embarked_map[x])
test["te_embarked"] = test["Embarked"].apply(lambda x: te_embarked_map[x])

#count_encording
ce_embarked_map = {"S":len(train[train["Embarked"]=="S"]),"C":len(train[train["Embarked"]=="C"]),"Q":len(train[train["Embarked"]=="Q"])}
train["ce_embarked"] = train["Embarked"].apply(lambda x: ce_embarked_map[x])
test["ce_embarked"] = test["Embarked"].apply(lambda x: ce_embarked_map[x])

#label_count_encoding
sorted_nums = np.sort(np.unique(train["ce_embarked"]))[::-1]
rank_map = {"{}".format(num) : i+1 for i,num in enumerate(sorted_nums)}
train["lce_embarked"] = train["ce_embarked"].apply(lambda x: rank_map[str(x)])
test["lce_embarked"] = test["ce_embarked"].apply(lambda x: rank_map[str(x)])

train.drop("Embarked",axis=1,inplace=True)
test.drop("Embarked",axis=1,inplace=True)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
cols = list(train.columns)
features = train[cols[2:]]
target = train[cols[1]]

features.to_csv("mod_train.csv",index=False)

ss = StandardScaler()
features_std = pd.DataFrame(ss.fit_transform(features),columns=cols[2:])
features_std.to_csv("std_mod_features.csv",index=False)

ms = MinMaxScaler()
features_mm = pd.DataFrame(ms.fit_transform(features),columns=cols[2:])
features_mm.to_csv("minmax_mod_features.csv",index=False)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from copy import copy
base_cols = cols[2:-3]
encoded_Emb_cols = cols[-3:]
scores = []

for col_name in encoded_Emb_cols:
    tmp_cols = copy(base_cols)
    tmp_cols.append(col_name)
    print(tmp_cols)
    clf = LogisticRegression()
    score = cross_val_score(clf, features_std[tmp_cols] , target, cv=3)
    scores.append(np.mean(score))

print(scores)
