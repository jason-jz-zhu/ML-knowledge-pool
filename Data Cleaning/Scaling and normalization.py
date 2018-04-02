from mlxtend.preprocessing import minmax_scaling

# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)

# Scaling (SVM, KNN)
# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])


# Normalization (t-tests, ANOVAs, linear regression, linear discriminant analysis and Gaussian naive Bayes)
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)


# reference
## https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data/notebook
