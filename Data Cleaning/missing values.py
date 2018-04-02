# Is this value missing becuase it wasn't recorded or becuase it dosen't exist?

# make copy to avoid changing original data (when Imputing)
new_data = original_data.copy()

# summzie the missing values
missing_values_count = new_data.isnull().sum()

# how many total missing values do we have?
total_cells = np.product(new_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100

# make new columns indicating what will be imputed
cols_with_missing = (col for col in new_data.columns
                                 if new_data[c].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()


# fill one value
subset_nfl_data.fillna(0)

# replace all NA's the value that comes directly after it in the same column,
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)

# Imputation
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
data_with_imputed_values = my_imputer.fit_transform(original_data)

# An Extension To Imputation
# make new columns indicating what will be imputed
cols_with_missing = (col for col in new_data.columns
                                 if new_data[c].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()

# Imputation
my_imputer = Imputer()
new_data = my_imputer.fit_transform(new_data)

# reference
## https://www.kaggle.com/jasonjzzhu88/data-cleaning-challenge-handling-missing-values/edit
## https://www.kaggle.com/dansbecker/handling-missing-values
