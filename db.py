import os
from dotenv import load_dotenv
from datetime import datetime
import psycopg2
import pytz

load_dotenv()

# Set timezone to Eastern Standard Time (EST)
est_timezone = pytz.timezone('America/New_York')

def get_db_connection():
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )
    return conn

def question_to_db(question: str, answer: str, sess_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS q_history (
        id SERIAL PRIMARY KEY,
        question TEXT,
        date DATE,
        time TIME,
        session_id TEXT,
        answer TEXT,
        UNIQUE(question, date, time, session_id, answer)
    )
    ''')

    row = []

    try:
        current_datetime = datetime.now(est_timezone)
        date = current_datetime.strftime('%Y-%m-%d')
        time = current_datetime.strftime('%H:%M:%S')
        row = (question, date, time, sess_id, answer)
        cursor.execute('''
                INSERT INTO q_history (question, date, time, session_id, answer)
                VALUES (%s, %s, %s, %s, %s)
                ''', row)
        conn.commit()  # Commit each successful insertion
    finally:
        # Close connection
        cursor.close()
        conn.close()