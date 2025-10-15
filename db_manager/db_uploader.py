import sqlite3
conn = sqlite3.connect("storage/file_monitor.db", check_same_thread=False)
cursor =conn.cursor()

def file_monitor_db(file_name):
    cursor.execute("INSERT INTO files (file_name) VALUES (?)", (file_name,))
    conn.commit()
