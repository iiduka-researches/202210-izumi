import os
from base64 import urlsafe_b64encode
import json
from pickle import load
from typing import Any, Dict, Optional, Sequence, Union
from email.header import Header
from email import encoders
from email.message import Message
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from googleapiclient.discovery import build
from utils.gmail.generate_token import generate
FilePath = Optional[Union[str, Sequence[str]]]
EXTENSION = Optional[Union[str, Sequence[str]]]

SCOPE = ['https://www.googleapis.com/auth/gmail.send']
TOKEN = 'utils/gmail/token.pickle'
SECRET = 'utils/gmail/client_secret.json'
ACCOUNT_JSON = './utils/gmail/account.json'


class GMailTransmitter:
    def __init__(self, service_account=None, password=None, user_id='me', encoding='utf-8',
                 token_path=TOKEN, secret_path=SECRET, account_json=ACCOUNT_JSON) -> None:
        credentials = get_credentials(token_path, secret_path)

        with open(account_json) as f:
            s = ''.join(f.readlines())
            account_info = json.loads(s)
        if service_account is None:
            service_account = account_info['service_account']
        if password is None:
            password = account_info['password']

        self.service = build('gmail', 'v1', credentials=credentials)
        self.user_id = user_id
        self.sender_account = service_account
        self.sender_password = password
        self.encoding = encoding

    def send(self, subject: str, body: str, to=None, file_path: FilePath = None, extension: EXTENSION = None,
             cc=None) -> None:
        message = self.create_message(subject, to, body, file_path, extension, cc)
        self.service.users().messages().send(userId=self.user_id, body=message).execute()

    def create_message(self, subject: str, recipient: str, body: str, file_path: FilePath, extension: EXTENSION,
                       cc: str) -> Dict[str, Any]:
        message = self.get_message_mime(subject, recipient, cc)
        message = attach_body(message, body)
        message = attach_file(message, file_path, extension)
        message = urlsafe_b64encode(message.as_bytes())
        return {'raw': message.decode()}

    def get_message_mime(self, subject: str, recipient: str, cc: str) -> Message:
        message = MIMEMultipart()
        message['Subject'] = Header(subject, self.encoding)
        message['To'] = recipient
        message['From'] = self.sender_account
        if cc:
            message['Cc'] = cc
        return message


def get_credentials(token_path: str, secret_path: str):
    if os.path.isfile(token_path):
        with open(token_path, 'rb') as token:
            return load(token)
    else:
        return generate(token_path, secret_path)


def attach_body(message: Message, body) -> Message:
    body = MIMEText(body)
    message.attach(body)
    return message


def get_extension(path: str) -> str:
    return str(os.path.splitext(path))


def attach_file(message: Message, path: FilePath, extension: EXTENSION) -> Message:
    if path:
        if extension:
            if isinstance(path, str) and isinstance(extension, str):
                _attach_file(message, path, extension)
            else:
                for p, e in zip(path, extension):
                    _attach_file(message, p, e)
        else:
            raise ValueError('Please set extension = {file}')
    return message


def _attach_file(message: Message, path: Optional[str], extension: Optional[str]) -> Message:
    if path and extension:
        attach = MIMEBase('application', extension)
        with open(path, 'rb') as f:
            attach.set_payload(f.read())
        encoders.encode_base64(attach)
        attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(path))
        message.attach(attach)
    return message
