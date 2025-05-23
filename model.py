import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("physics_qna.csv")

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data['question'])


model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, data['answer'])

while True:
    user_question = input("Ask a Physics question (or type 'exit' to quit): ")

    if user_question.lower() == 'exit':
        print("THANKS FOR USING UTSAV'S AI")
        break

    question_vector = vectorizer.transform([user_question])
    predicted_answer = model.predict(question_vector)[0]

    print("Answer:", predicted_answer)





