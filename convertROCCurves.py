
## Load Annotation Classifier Data
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


## Load Data
AC_df=pd.read_csv('Annotation_Classifier.csv')
SC_df=pd.read_csv('Sequence_Classifier_Data.csv')
EC_df=pd.read_csv('Ensemble_Classifier.csv')
StructClass_df=pd.read_csv('Structure_Classifier.csv')
SC_fixed_df=pd.read_csv('Sequence_Classifier_Fixed.csv')
Struct_test=pd.read_csv('Structure_Classifier_Test.csv')

## Load Test Data
testIDs=pd.read_csv('/Users/trayneli/Projects/CompBioProjectData/Test_IDs.txt', header=None)

## Load Sequence Classifier Data
all_values=SC_fixed_df["EC"].tolist() +  AC_df["EC"].tolist() + AC_df["Pred_EC"].tolist() + SC_fixed_df["Pred_EC"].tolist() + EC_df["Pred_EC"].tolist() + EC_df["EC"].tolist() + Struct_test["Pred_EC"].tolist() + Struct_test["EC"].tolist()
DistinctValues = set(all_values)
EC2idx = {ec: idx for idx, ec in enumerate(DistinctValues)}

AC_df['EC_Pred_Mapped'] = AC_df['Pred_EC'].map(EC2idx)
AC_df['EC_Actual'] = AC_df['EC'].map(EC2idx)

SC_fixed_df['SC_Pred_Mapped'] = SC_fixed_df['Pred_EC'].map(EC2idx)
SC_fixed_df['SC_Actual'] = SC_fixed_df['EC'].map(EC2idx)

EC_df['EC_Pred_Mapped'] = EC_df['Pred_EC'].map(EC2idx)
EC_df['EC_Actual'] = EC_df['EC'].map(EC2idx)

Struct_test['StrC_Pred_Mapped'] = Struct_test['Pred_EC'].map(EC2idx)
Struct_test['StrC_Actual'] = Struct_test['EC'].map(EC2idx)


def GetMaxValue(input_row):
    raw_string=input_row.strip("'")
    embedding_nums = [float(num) for num in raw_string.strip('[]').split(', ')]
    exp_values = np.exp(embedding_nums)
    probabilities=exp_values / np.sum(exp_values)
    MaxProb=max(probabilities)
    return MaxProb


def add_column_prob(input_df):
    ProbabilityList=[]
    for index, row in input_df.iterrows():
        ProbabilityList.append(GetMaxValue(row['Weights']))
    input_df['ProbabilityOutput']=ProbabilityList
    return input_df


StructProb=add_column_prob(Struct_test)
SCProb=add_column_prob(SC_fixed_df)
ECProb=add_column_prob(EC_df)
ACProb=add_column_prob(AC_df)


## Filter for test Data
filtered_df_ac = AC_df[AC_df['UniProtID'].isin(testIDs[0])]
filtered_SC_fixed_df = SC_fixed_df[SC_fixed_df['UniProtID'].isin(testIDs[0])]
#filtered_df_struct = StructClass_df[Struct_test['UniProtID'].isin(testIDs[0])]
filtered_df_ec = EC_df[EC_df['UniProtID'].isin(testIDs[0])]

## Set True and false values
AC_true=filtered_df_ac['EC_Actual']
AC_pred=filtered_df_ac['EC_Pred_Mapped'] 
StrC_true=filtered_SC_fixed_df['SC_Actual']
StrC_pred=filtered_SC_fixed_df['SC_Pred_Mapped'] 
EC_true=EC_df['EC_Actual']
EC_pred=EC_df['EC_Pred_Mapped']
SC_true=Struct_test['StrC_Actual']
SC_pred=Struct_test['StrC_Pred_Mapped']


models = [
    ("Ensemble Model Performance", EC_true, EC_pred),
    ("Annotation Classification Model Performance", AC_true, AC_pred),
    ("Structural Classification Model Performance", SC_true, SC_pred),
    ("Sequence Classification Model Performance", StrC_true, StrC_pred)
]

colors = sns.color_palette("muted", 4) 
fig, axes = plt.subplots(2, 2, figsize=(18, 12))  # 2x2 grid
axes = axes.flatten()  # Make it easy to loop

for idx, (Title, true_val, pred_val) in enumerate(models):
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(true_val, pred_val),
        'Precision Macro': precision_score(true_val, pred_val, average='macro'),
        'Precision Micro': precision_score(true_val, pred_val, average='micro'),
        'Precision Weighted': precision_score(true_val, pred_val, average='weighted'),
        'Recall Macro': recall_score(true_val, pred_val, average='macro'),
        'Recall Micro': recall_score(true_val, pred_val, average='micro'),
        'Recall Weighted': recall_score(true_val, pred_val, average='weighted'),
        'F1 Macro': f1_score(true_val, pred_val, average='macro'),
        'F1 Micro': f1_score(true_val, pred_val, average='micro'),
        'F1 Weighted': f1_score(true_val, pred_val, average='weighted'),
    }

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    # Plot
    ax = axes[idx]
    bars = ax.barh(metric_names, metric_values, color=colors[idx])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Score')
    ax.set_title(Title)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', va='center')

plt.tight_layout()
plt.show()


import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Step 2: Create the "true" binary labels
# 1 if predicted = actual, 0 otherwise
EC_df['TrueLabel'] = (EC_df['EC_Pred_Mapped'] == EC_df['EC_Actual']).astype(int)
Struct_test['TrueLabel'] = (Struct_test['StrC_Pred_Mapped'] == Struct_test['StrC_Actual']).astype(int)
filtered_df_ac['TrueLabel'] = (filtered_df_ac['EC_Pred_Mapped'] == filtered_df_ac['EC_Actual']).astype(int)
filtered_SC_fixed_df['TrueLabel'] = (filtered_SC_fixed_df['SC_Pred_Mapped'] == filtered_SC_fixed_df['SC_Actual']).astype(int)


# Step 3: Get fpr, tpr for ROC
fpr1, tpr1, thresholds = roc_curve(EC_df['TrueLabel'], EC_df['ProbabilityOutput'])
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, thresholds = roc_curve(Struct_test['TrueLabel'], Struct_test['ProbabilityOutput'])
roc_auc2 = auc(fpr2, tpr2)

fpr3, tpr3, thresholds = roc_curve(filtered_df_ac['TrueLabel'], filtered_df_ac['ProbabilityOutput'])
roc_auc3 = auc(fpr3, tpr3)

fpr4, tpr4, thresholds = roc_curve(filtered_SC_fixed_df['TrueLabel'], filtered_SC_fixed_df['ProbabilityOutput'])
roc_auc4 = auc(fpr4, tpr4)



# Assume you already have your original fpr, tpr, and roc_auc

# Now, plot all
plt.figure()

plt.plot(fpr1, tpr1, color='blue', lw=2, label=f'Ensemble Classifier (AUC = {roc_auc1:.2f})')
plt.plot(fpr2, tpr2, color='green', lw=2, label=f'Structural Classifier (AUC = {roc_auc2:.2f})')
plt.plot(fpr3, tpr3, color='red', lw=2, label=f'Annotation Classifier (AUC = {roc_auc3:.2f})')
plt.plot(fpr4, tpr4, color='purple', lw=2, label=f'Sequence Classifier(AUC = {roc_auc4:.2f})')

# Random classifier line
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Beautify the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC Curves')
plt.legend(loc="lower right")
plt.grid()
plt.show()

