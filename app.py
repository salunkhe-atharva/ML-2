import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_classifier():
    # Dataset
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

    # Feature Extraction
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1
    )
    X = vectorizer.fit_transform(emails)

    # Split data (stratify ensures balanced spam/ham in train and test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=42, stratify=labels
    )

    # Train SVM Model
    svm_model = LinearSVC(C=1.0, dual="auto")
    svm_model.fit(X_train, y_train)

    # Evaluate
    y_pred = svm_model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # User Input
    print("\n--- Spam Detection System ---")
    new_email = input("Enter a new email message: ")
    if not new_email.strip():
        print("Empty input. Exiting.")
        return

    new_email_vectorized = vectorizer.transform([new_email])
    prediction = svm_model.predict(new_email_vectorized)

    result = "SPAM" if prediction[0] == 1 else "NOT SPAM (Ham)"
    print(f"Result: The email is {result}.")

if __name__ == "__main__":
    run_classifier()
