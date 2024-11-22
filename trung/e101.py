import pandas as pd
from sklearn.naive_bayes import CategoricalNB

# Creating the dataset
data = {
    'Age': [25, 20, 25, 45, 20, 25],
    'Car': ['sports', 'vintage', 'sports', 'suv', 'sports', 'suv'],
    'Class': ['L', 'H', 'L', 'H', 'L', 'H']
}

df = pd.DataFrame(data)

# Mapping categorical data to numerical values for naive Bayes
age_mapping = {20: 0, 25: 1, 45: 2}
car_mapping = {'sports': 0, 'vintage': 1, 'suv': 2, 'truck': 3}
class_mapping = {'L': 0, 'H': 1}

df['Age'] = df['Age'].map(age_mapping)
df['Car'] = df['Car'].map(car_mapping)
df['Class'] = df['Class'].map(class_mapping)

X = df[['Age', 'Car']]
y = df['Class']

# Training the Naive Bayes classifier
model = CategoricalNB()
model.fit(X, y)

# New data point
new_data = pd.DataFrame([[1, 3]], columns=['Age', 'Car']) # Age=23 mapped to 1 (closer to 25), Car=truck mapped to 3

# Prediction using Naive Bayes
naive_bayes_prediction = model.predict(new_data)
naive_bayes_class = 'L' if naive_bayes_prediction[0] == 0 else 'H'

print(f"Naive Bayes classification: {naive_bayes_class}")

# For Full Bayes Approach (Manually calculate probabilities)
prob_L = (3/6)  # Prior probability of Class L
prob_H = (3/6)  # Prior probability of Class H

# Likelihood of Age=23 and Car=truck given Class L
likelihood_L_age = 2/3  # 2 out of 3 for Age close to 25
likelihood_L_car = 0    # 0 out of 3 for Car as truck in Class L
likelihood_L = likelihood_L_age * likelihood_L_car

# Likelihood of Age=23 and Car=truck given Class H
likelihood_H_age = 1/3  # 1 out of 3 for Age close to 25 (no close mapping to 20 or 45)
likelihood_H_car = 0    # 0 out of 3 for Car as truck in Class H
likelihood_H = likelihood_H_age * likelihood_H_car

# Posterior probability calculation
posterior_L = prob_L * likelihood_L
posterior_H = prob_H * likelihood_H

# Classifying based on higher posterior probability
if posterior_L > posterior_H:
    full_bayes_class = 'L'
else:
    full_bayes_class = 'H'

print(f"Full Bayes classification: {full_bayes_class}")
