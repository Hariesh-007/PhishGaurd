import sqlite3
import os

db_path = 'instance/phishing_detector.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Checking 'User' table...")
    cursor.execute("PRAGMA table_info(User)")
    columns = cursor.fetchall()
    for col in columns:
        print(col)
        
    print("\nChecking 'EmailLog' table...")
    cursor.execute("PRAGMA table_info(EmailLog)")
    columns = cursor.fetchall()
    for col in columns:
        print(col)
    
    conn.close()
else:
    print(f"{db_path} does not exist.")
