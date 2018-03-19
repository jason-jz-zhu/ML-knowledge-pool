# make copy to avoid changing original data (when Imputing)
new_data = original_data.copy()

# make new columns indicating what will be imputed
cols_with_missing = (col for col in new_data.columns
                                 if new_data[c].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()

# Imputation
my_imputer = Imputer()
new_data = my_imputer.fit_transform(new_data)
