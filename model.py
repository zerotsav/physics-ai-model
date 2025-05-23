# Step 1: Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Step 2: Load the physics QnA dataset from a CSV file
data = pd.read_csv("physics_qna.csv")

# Step 3: Separate the questions and answers
X = data['question']  # Questions will be the input
y = data['answer']    # Answers are what we want to predict

# Step 4: Convert text questions into numerical data using TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Step 5: Create and train a K-Nearest Neighbors model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_vectorized, y)

# Step 6: Make a prediction for a new question
question = "What is Newton's second law?"
question_vector = vectorizer.transform([question])
prediction = model.predict(question_vector)

# Step 7: Show the predicted answer
print("Predicted answer:", prediction[0])
