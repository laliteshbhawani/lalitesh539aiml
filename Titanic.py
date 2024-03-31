import numpy as np
import pandas as pd
from faker import Faker

# Initialize Faker to generate fake data
fake = Faker()

# Generate random data points for analysis
np.random.seed(0)
num_passengers = 1000

# Generating random features: Age, Fare, Sex, Passenger Class, and Number of Siblings/Spouses Aboard
age_values = np.random.randint(1, 100, size=num_passengers)
fare_values = np.random.uniform(0, 500, size=num_passengers)
sex_values = np.random.randint(0, 2, size=num_passengers)  # 0 for female, 1 for male
class_values = np.random.randint(1, 4, size=num_passengers)  # Passenger class (1, 2, or 3)
sibsp_values = np.random.randint(0, 9, size=num_passengers)  # Number of Siblings/Spouses Aboard

# Generating random names
names = [fake.name() for _ in range(num_passengers)]

# Generating random survival status (0 for not survived, 1 for survived)
survival_weights = np.array([0.01, 0.02, 0.5, 0.1, 0.05])  # Weight parameters
bias_value = -3  # Bias parameter

# Calculate probabilities of survival using logistic function
logits = (
    survival_weights[0] * age_values +
    survival_weights[1] * fare_values +
    survival_weights[2] * sex_values +
    survival_weights[3] * class_values +
    survival_weights[4] * sibsp_values +
    bias_value
)
probabilities = 1 / (1 + np.exp(-logits))

# Randomly assign survival status based on probabilities
survived_values = np.random.binomial(n=1, p=probabilities)

# Combine features and survival status into a DataFrame
passenger_data = pd.DataFrame({
    'Name': names,
    'Age': age_values,
    'Fare': fare_values,
    'Sex': sex_values,
    'Class': class_values,
    'SibSp': sibsp_values,
    'Survived': survived_values
})

# Save data to CSV file
passenger_data.to_csv('passenger_data.csv', index=False)
