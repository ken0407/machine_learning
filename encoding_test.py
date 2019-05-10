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

train_X,train_y,test_X,test_y = train_test_split(train[cols[2:]],train[cols[1]],test_size=0.3,random_state=1234)

train_X.to_csv("mod_train.csv",index=False)

ss = StandardScaler()
train_X_std = pd.DataFrame(ss.fit_transform(train_X),columns=cols[2:])
train_X_std.to_csv("std_mod_train.csv",index=False)

ms = MinMaxScaler()
train_X_mm = pd.DataFrame(ms.fit_transform(train_X),columns=cols[2:])
train_X_mm.to_csv("minmax_mod_train.csv",index=False)
