from pickle import dump
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPE = ['https://www.googleapis.com/auth/gmail.send']


def generate(token_path: str, secret_path: str) -> None:
    flow = InstalledAppFlow.from_client_secrets_file(secret_path, SCOPE)
    credentials = flow.run_console()
    with open(token_path, "wb") as token:
        dump(credentials, token)
    return credentials
