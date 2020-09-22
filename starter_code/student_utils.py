import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools 

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    joined = pd.merge(df, ndc_df[['Non-proprietary Name', 'NDC_Code']], how='left', 
                      left_on='ndc_code', right_on='NDC_Code')
    joined['generic_drug_name'] = joined['Non-proprietary Name']
    joined = joined.drop(['NDC_Code', 'Non-proprietary Name', 'ndc_code'], axis=1)
    return joined

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    
    df = df.sort_values('encounter_id')
    # group same patients together and select their first encounter
    first_encounter_values = df.groupby('patient_nbr')['encounter_id'].head(1).values
    
    # select all first encounters for patients from the dataframe and return it
    return df[df['encounter_id'].isin(first_encounter_values)]


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    
    train_percent = 0.6
    valid_percent = 0.2
    test_percent = 0.2
    
    df = df.iloc[np.random.permutation(len(df))] # permute and select
    unique_patients = df[patient_key].unique()
    num_unique = len(unique_patients)
    
    train_size = round(num_unique * train_percent)
    valid_size = round(num_unique * valid_percent)
    test_size = round(num_unique * test_percent)

    train = df[df[patient_key].isin(unique_patients[:train_size])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_patients[train_size:train_size+valid_size])].reset_index(drop=True)
    test = df[df[patient_key].isin(unique_patients[train_size+valid_size:])].reset_index(drop=True)

    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        
        tf_categorical_feature_column_vocab = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file=vocab_file_path, num_oov_buckets=1)
        
        tf_categorical_feature_indicator = tf.feature_column.indicator_column(tf_categorical_feature_column_vocab)
        
        output_tf_list.append(tf_categorical_feature_indicator)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    
    normalization_func = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col, default_value=default_value, 
                                                          normalizer_fn=normalization_func, dtype=tf.float64)
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    
    df['out'] = np.where(df[col] >= 5, 1, 0)
    return df.out.values
