import os
from dotenv import load_dotenv
import requests

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.getenv["TELEGRAM_CHAT_ID"]

def send_telegram_message(text='Cell execution completed.'):
    requests.post( f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage', 
        params=dict(chat_id=TELEGRAM_CHAT_ID, text=text)