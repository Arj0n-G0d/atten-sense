import sqlite3
from datetime import date

def create_db():
    # Connect to SQLite database (creates file if it doesn't exist)
    conn = sqlite3.connect("atten_sense.db")
    cursor = conn.cursor()

    # Table for focus sessions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS focus_sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        date TEXT NOT NULL
    )
    """)

    # Table for individual focus logs linked to a session
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS focus_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        start TEXT NOT NULL,
        end TEXT NOT NULL,
        focus_state TEXT NOT NULL,  -- Options: 'Focused', 'Not Focused'
        FOREIGN KEY (session_id) REFERENCES focus_sessions (session_id)
    )
    """)

    conn.commit()
    conn.close()

def insert_focus_session(name):
    conn = sqlite3.connect("atten_sense.db")
    cursor = conn.cursor()

    today = date.today().isoformat()

    # Insert session and get the ID
    cursor.execute("INSERT INTO focus_sessions (name, date) VALUES (?, CURRENT_TIMESTAMP)", (name,))
    session_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return session_id

def insert_focus_log(session_id, start, end, focus_state):
    conn = sqlite3.connect("atten_sense.db")
    cursor = conn.cursor()
    
    if focus_state == 0 :
        focus_state = "Not Focused"
    else :
        focus_state = "Focused"

    cursor.execute("""
        INSERT INTO focus_logs (session_id, start, end, focus_state)
        VALUES (?, ?, ?, ?)
    """, (session_id, start, end, focus_state))

    conn.commit()
    conn.close()
