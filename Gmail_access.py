import imaplib
import email
from email.header import decode_header
import pandas as pd
from datetime import datetime
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Function to clean the text for creating a valid filename
def clean_text(text):
    return "".join(c if c.isalnum() else "_" for c in text)

# Connect to the Gmail IMAP server
def connect_to_gmail(username, app_password):
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(username, app_password)
    return mail

# Fetch all emails from the Spam folder
def fetch_emails(mail):
    mail.select("[Gmail]/Spam")
    result, data = mail.search(None, "ALL")
    email_ids = data[0].split()
    return email_ids

# Predict whether the email body is spam
def predict_spam(model, vectorizer, body):
    body_vector = vectorizer.transform([body])
    return model.predict(body_vector)[0]

# Extract email details and append to a list
def extract_emails(mail, email_ids, model, vectorizer):
    emails_data = []
    for email_id in email_ids:
        result, msg_data = mail.fetch(email_id, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        # Decode email date
        date_ = msg.get("Date")
        
        # Extract year from the date
        try:
            parsed_date = datetime.strptime(date_, "%a, %d %b %Y %H:%M:%S %z")
            year = parsed_date.year
        except ValueError:
            # Handle date formats that might not match the expected format
            year = None

        # Accept emails from 2024 only
        if year != 2024:
            continue

        # Decode email subject
        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding if encoding else "utf-8")
        
        # Decode email from
        from_ = msg.get("From")

        # Extract the email message
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                # Skip any attachments
                if "attachment" in content_disposition:
                    continue
                
                if content_type == "text/plain":
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
                elif content_type == "text/html":
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')

        # Predict if the email is spam
        is_spam = predict_spam(model, vectorizer, body)
        spam_status = "Spam" if is_spam else "Not Spam"
        print(date_, spam_status)
        
        emails_data.append({
            "Subject": subject,
            "From": from_,
            "Date": date_,
            "Spam Status": spam_status
        })

    return emails_data

# Save the emails data to a CSV file
def convert_to_df(emails_data):
    df = pd.DataFrame(emails_data)
    return df
    # df.to_csv(filename, index=False)

# Main function to perform the email extraction
def main():
    username = "Enter your email"
    app_password = "Enter your password"  # Use an app password if 2FA is enabled

    # Load the model and vectorizer
    with open('models/nb_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('models/count_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    print("Model and vectorizer loaded successfully.")

    mail = connect_to_gmail(username, app_password)
    email_ids = fetch_emails(mail)
    emails_data = extract_emails(mail, email_ids, model, vectorizer)
    print(convert_to_df( emails_data))

if __name__ == "__main__":
    main()
