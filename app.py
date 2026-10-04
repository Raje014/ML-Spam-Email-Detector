import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# load dataset
df = pd.read_csv("emails.csv")

# input target
X = df["text"]
Y = df["spam"]

# train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# TF-IDF
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# NAIVE BAYES MODEL

model = MultinomialNB(alpha=0.1)
model.fit(X_train_tfidf, Y_train)


# test model
predictions = model.predict(X_test_tfidf)


# model performance
accuracy = accuracy_score(Y_test, predictions)

conf_matrix = confusion_matrix(Y_test, predictions)

report = classification_report(
    Y_test,
    predictions,
    output_dict=True
)


# streamlit UI
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Detector")

st.write(
    "Enter an email below and the Machine Learning model "
    "will predict whether it is Spam or Not Spam."
)


# email input
email = st.text_area(
    "Enter your email:",
    height=200,
    placeholder="Example: Congratulations! You have won a prize..."
)

# prediction
if st.button("🔍 Check Email"):

    if email.strip() == "":
        
        st.warning("⚠️ Please enter an email.")

    else:

        # Convert email into TF-IDF numbers
        email_tfidf = vectorizer.transform([email])

        # Prediction
        prediction = model.predict(email_tfidf)[0]

        # Probability
        probabilities = model.predict_proba(email_tfidf)[0]

        # Find probability of spam (class = 1)
        spam_index = list(model.classes_).index(1)

        spam_probability = probabilities[spam_index]

        # result
        if prediction == 1:
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ NOT SPAM")

        # probability
        st.write(
            f"### Spam Probability: "
            f"{spam_probability * 100:.2f}%"
        )

        st.progress(float(spam_probability))

# model performance
st.divider()

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with col2:
    st.metric(
        "Spam Precision",
        f"{report['1']['precision'] * 100:.2f}%"
    )


with col3:
    st.metric(
        "Spam Recall",
        f"{report['1']['recall'] * 100:.2f}%"
    )


with col4:
    st.metric(
        "Spam F1",
        f"{report['1']['f1-score'] * 100:.2f}%"
    )


# confusion matrics
st.subheader("🔍 Confusion Matrix")

st.write(
    "Rows = Actual | Columns = Predicted"
)

st.dataframe(
    pd.DataFrame(
        conf_matrix,
        index=["Actual Not Spam", "Actual Spam"],
        columns=["Predicted Not Spam", "Predicted Spam"]
    )
)


# classification report

st.subheader("📋 Classification Report")

st.dataframe(
    pd.DataFrame(report).transpose()
)