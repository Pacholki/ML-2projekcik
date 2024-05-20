import numpy as np

def remove_outliers(df=None, columns=None):


    if columns == None:
        columns = df.select_dtypes(include=[np.number]).columns.to_list()

    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

    return df
