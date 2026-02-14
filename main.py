# Updated main.py

# Session ID management
import uuid
from datetime import datetime

class Session:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()

    def __repr__(self):
        return f"Session(session_id={self.session_id}, timestamp={self.timestamp})"

# Example of session creation
session = Session()
print(session)

# Persistent PDF storage in stored_pdfs directory
import os

STORED_PDFS_DIR = 'stored_pdfs'

if not os.path.exists(STORED_PDFS_DIR):
    os.makedirs(STORED_PDFS_DIR)

# Function to save PDF
def save_pdf(file_name, content):
    path = os.path.join(STORED_PDFS_DIR, file_name)
    with open(path, 'wb') as f:
        f.write(content)

# Example usage
# save_pdf('sample.pdf', b'PDF binary content here')
