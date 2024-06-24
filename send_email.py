import boto3
import os


def send_email(subject, body, recipient):
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID").strip()
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY").strip()
    aws_region = os.getenv("AWS_REGION").strip()
    ses = boto3.client(
        "ses",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # The email sender and recipient
    sender = os.getenv("EMAIL").strip()

    try:
        # Send the email
        response = ses.send_email(
            Source=sender,
            Destination={
                "ToAddresses": [
                    recipient,
                ],
            },
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
            },
        )
        print(f"Email sent! Message ID: {response['MessageId']}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    subject = "SLURM Job Notification"
    body = "All SLURM jobs have completed."
    recipient = os.getenv("EMAIL").strip()

    send_email(subject, body, recipient)
