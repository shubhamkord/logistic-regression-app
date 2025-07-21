
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

# Dummy data and model training for demonstration (replace with your real training code)
X = np.array([[1, 22, 1, 71.2833], [3, 26, 0, 7.925], [1, 35, 1, 53.1]])
y = [1, 0, 1]

model = LogisticRegression()
model.fit(X, y)

# Save model
with open('logreg_model.pkl', 'wb') as f:
    pickle.dump(model, f)
