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

    # Get date
    cursor.execute("SELECT date FROM focus_sessions WHERE session_id = ?", (session_id,))
    session_date = cursor.fetchone()[0]

    conn.commit()
    conn.close()
    return session_id, session_date

def insert_focus_log(session_id, start, end, focus_state):
    conn = sqlite3.connect("atten_sense.db")
    cursor = conn.cursor()
    
    if isinstance(focus_state, bool):
        focus_state = "Focused" if focus_state else "Not Focused"
    elif focus_state == 0:
        focus_state = "Not Focused"
    elif focus_state == 1:
        focus_state = "Focused"
        
    cursor.execute("""
        INSERT INTO focus_logs (session_id, start, end, focus_state)
        VALUES (?, ?, ?, ?)
    """, (session_id, start, end, focus_state))
    conn.commit()
    conn.close()

def get_session_by_id(session_id, include_logs=False):
    """Retrieves a session by its ID, optionally including its focus logs"""
    conn = sqlite3.connect("atten_sense.db")
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    # Get the session details
    cursor.execute("""
        SELECT session_id, name, date 
        FROM focus_sessions 
        WHERE session_id = ?
    """, (session_id,))
    
    session = cursor.fetchone()
    
    if not session:
        conn.close()
        return None
    
    # Convert to dict for easier manipulation
    session_dict = dict(session)
    
    if include_logs:
        # Get the focus logs for this session
        cursor.execute("""
            SELECT start, end, 
                   focus_state
            FROM focus_logs
            WHERE session_id = ?
            ORDER BY start
        """, (session_id,))
        
        # Convert focus_state to boolean
        logs = []
        for log in cursor.fetchall():
            logs.append((log['start'], log['end'], log['focus_state'] == 'Focused'))
            
        session_dict['logs'] = logs
    
    conn.close()
    return session_dict

def get_sessions_by_name(name):
    """Retrieves all sessions for a given name"""
    conn = sqlite3.connect("atten_sense.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT session_id, name, date
        FROM focus_sessions
        WHERE name LIKE ?
        ORDER BY date DESC
    """, (f"%{name}%",))  # Using LIKE for partial matching
    
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return sessions