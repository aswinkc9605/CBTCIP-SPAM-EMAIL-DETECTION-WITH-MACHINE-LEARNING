import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

file_path = 'Spam.csv'
data = pd.read_csv(file_path)

data_cleaned = data[['v1', 'v2']]
data_cleaned.columns = ['label', 'text']

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data_cleaned['text'] = data_cleaned['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data_cleaned['text'])

y = data_cleaned['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

def classify_email(email, model, vectorizer):
    email_processed = preprocess_text(email)
    email_vectorized = vectorizer.transform([email_processed])
    prediction = model.predict(email_vectorized)
    return prediction[0]

spam_emails = []
non_spam_emails = []

for index, row in data_cleaned.iterrows():
    label = classify_email(row['text'], model, vectorizer)
    if label == 'spam':
        spam_emails.append(row['text'])
    else:
        non_spam_emails.append(row['text'])

with open('spam_emails.txt', 'w', encoding='utf-8') as f:
    for email in spam_emails:
        f.write(email + '\n')

with open('non_spam_emails.txt', 'w', encoding='utf-8') as f:
    for email in non_spam_emails:
        f.write(email + '\n')

print(f"Total spam emails: {len(spam_emails)}")
print(f"Total non-spam emails: {len(non_spam_emails)}")

