import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



NUMERICAL_VARIBLES = ['loan_amount', 'rate_of_interest', 'Upfront_charges', 'property_value', 'income', 'LTV', 'dtir1']
BINARY_VARIBLES = ['loan_limit', 'approv_in_adv', 'Credit_Worthiness', 'open_credit', 'business_or_commercial', 'Neg_ammortization',
                   'interest_only', 'lump_sum_payment', 'construction_type', 'Secured_by', 'co-applicant_credit_type', 'age', 'Security_Type']

CATEGORICAL_VARIBLES = ['Gender', 'loan_type', 'loan_purpose', 'term', 'occupancy_type', 'total_units', 'credit_type', 'Region']

TARGET = ['Status']

FEATURES = NUMERICAL_VARIBLES + BINARY_VARIBLES + CATEGORICAL_VARIBLES + TARGET



transformer = pickle.load(open('artifacts/transformer.pkl', 'rb'))


age_range_dict = {
    "<25": 1,
    "25-34": 2,
    "35-44": 3,
    "45-54": 4,
    "55-64": 5,
    "65-74": 6,
    ">74": 7
}

def pre_processing(X, fill_na = False):
    
    # Processing Numberical Varibles
    for num_col in NUMERICAL_VARIBLES:
        if fill_na:
            group_means = X.groupby('Status')[num_col].transform('mean')
            X.loc[:, num_col] = X.loc[:, num_col].fillna(group_means)
        X.loc[:,num_col] = transformer[num_col].transform(X[num_col].values.reshape(-1, 1))
    
    
    # Processing Categorical Varibles
    for cat_col in CATEGORICAL_VARIBLES:
        if fill_na:
            X.loc[:, cat_col] = X.loc[:, cat_col].fillna(X[cat_col].mode()[0])
        encoded_features = transformer[cat_col].transform(X[cat_col].values.reshape(-1,1)).toarray()
        encoded_df = pd.DataFrame(encoded_features, columns=transformer[cat_col].get_feature_names_out([cat_col]))
        X = X.reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)
        X = pd.concat([X, encoded_df], axis=1)
        X = X.drop(columns=[cat_col])
    
    
    # Processing Binary Varibles
    for bin_col in BINARY_VARIBLES:
        if fill_na:
            X.loc[:, bin_col] = X.loc[:, bin_col].fillna(X[bin_col].mode()[0])
            X[bin_col] = X[bin_col].fillna(X[bin_col].mode()[0])
        if bin_col != 'age':
            X[bin_col]= transformer[bin_col].transform(X[bin_col])
        
    
    # Preprocessing Age
    X['age'] = X['age'].replace(age_range_dict)
    
    return X





def plot_confusion_matrix(true_labels, predicted_labels):
    
    cnf_mat = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cnf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()    


def print_metrics(y_true, y_pred):
    print('Accuracy: {0}'.format(accuracy_score(y_true, y_pred)))
    
    print('*********************')
    print('Precision: {0}'.format(precision_score(y_true, y_pred)))
    
    print('*********************')
    print('Recall: {0}'.format(recall_score(y_true, y_pred)))
    
    print('*********************')
    print('F1-Score: {0}'.format(f1_score(y_true, y_pred)))
    
    
        