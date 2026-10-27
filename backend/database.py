import sqlite3
import os
from datetime import datetime

#DB file path : backend folder ke andr mes_production.db
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mes_production.db")

def get_connection():
    """Returns a SQLite db connection with multithread support"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(): 
    """DB and Table initialization"""
    conn = get_connection()
    cursor = conn.cursor()

# 1. Running totals table 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS """)