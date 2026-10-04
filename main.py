import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('emails.csv')

# print(df.head())

X = df["text"]
Y = df["spam"]

# print(X.head())
# print(Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# print(len(X_train))
# print(len(X_test))

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = MultinomialNB()

model.fit(X_train_tfidf, Y_train)

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(Y_test, predictions)

new_email = input("Enter an email: ")

new_email_tfidf = vectorizer.transform([new_email])

prediction = model.predict(new_email_tfidf)

if prediction[0] == 1:
    print("🚨 Spam")
else:
    print("✅ Not Spam")