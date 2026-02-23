import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@st.cache_resource
def train_model():
   
    emails = [
        "Congratulations! You’ve won a free iPhone", "Claim your lottery prize now",
        "Exclusive deal just for you", "Act fast! Limited-time offer",
        "Click here to secure your reward", "Win cash prizes instantly by signing up",
        "Limited-time discount on luxury watches", "Get rich quick with this secret method",
        "Hello, how are you today", "Please find the attached report",
        "Thank you for your support", "The project deadline is next week",
        "Can we reschedule the meeting to tomorrow", "Your invoice for last month is attached",
        "Looking forward to our call later today", "Don’t forget the team lunch tomorrow",
        "Meeting agenda has been updated", "Here are the notes from yesterday’s discussion",
        "Please confirm your attendance for the workshop", "Let’s finalize the budget proposal by Friday"
    ]
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1
    )
    X = vectorizer.fit_transform(emails)

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=42, stratify=labels
    )

    svm_model = LinearSVC(C=1.0, dual="auto")
    svm_model.fit(X_train, y_train)
    
    y_pred = svm_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    
    return vectorizer, svm_model, acc

st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("📧 Spam Detection System")

vectorizer, svm_model, accuracy = train_model()

st.sidebar.write(f"**Model Accuracy:** {accuracy:.2f}%")
st.sidebar.write("Algorithm: Linear SVC")

st.write("Enter an email message below to check if it's Spam or Ham.")
user_input = st.text_input("Email Message:", placeholder="e.g., You won a prize!")

if st.button("Predict"):
    if user_input.strip():
        new_email_vectorized = vectorizer.transform([user_input])
        prediction = svm_model.predict(new_email_vectorized)

        if prediction[0] == 1:
            st.error("🚨 Result: This looks like SPAM!")
        else:
            st.success("✅ Result: This is NOT SPAM (Ham).")
    else:
        st.warning("Please enter some text first.")
