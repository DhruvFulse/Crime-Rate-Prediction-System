"""Small SQLite layer: accounts, login history and user-registered crime reports.

The database file (crime_system.db) is created automatically on first run.
Default administrator account: username `admin`, password `Admin@123`
(change it after the first login by adding a new admin from the Users page).
"""
import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(os.environ.get("CRIME_DB_PATH", Path(__file__).with_name("crime_system.db")))

DATA_COLUMNS = ["Record_ID", "City", "Area_Type", "Year", "Month", "Day_Type",
                "Crime_Type", "Time_of_Day", "Population_Density", "Unemployment_Rate",
                "Literacy_Rate", "Police_Per_100k", "CCTV_Coverage", "Street_Lighting",
                "Avg_Income_K", "Prior_Incidents", "Crime_Rate", "Risk_Level"]

STATUSES = ["Registered", "Under Review", "Under Investigation", "Resolved", "Closed"]

DEFAULT_ADMIN = ("admin", "Admin@123", "System Administrator")


def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _hash(password, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 120_000).hex()


def init_db():
    with _conn() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, full_name TEXT, role TEXT NOT NULL,
            salt TEXT NOT NULL, password_hash TEXT NOT NULL, created_at TEXT)""")
        con.execute("""CREATE TABLE IF NOT EXISTS login_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, role TEXT,
            event TEXT, timestamp TEXT)""")
        cols = ", ".join(f"{c} TEXT" if c in ("Record_ID", "City", "Area_Type", "Day_Type", "Crime_Type",
                                              "Time_of_Day", "Risk_Level") else f"{c} REAL"
                         for c in DATA_COLUMNS)
        con.execute(f"""CREATE TABLE IF NOT EXISTS crime_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT, reported_by TEXT, reported_at TEXT,
            notes TEXT, {cols})""")
        con.execute("""CREATE TABLE IF NOT EXISTS status_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT, status TEXT,
            changed_by TEXT, remarks TEXT, timestamp TEXT)""")
        # databases created before the status feature: add the column and back-fill history
        existing = [r["name"] for r in con.execute("PRAGMA table_info(crime_reports)")]
        if "status" not in existing:
            con.execute("ALTER TABLE crime_reports ADD COLUMN status TEXT DEFAULT 'Registered'")
        con.execute("UPDATE crime_reports SET status='Registered' WHERE status IS NULL")
        con.execute("""INSERT INTO status_history (record_id, status, changed_by, remarks, timestamp)
            SELECT Record_ID, 'Registered', reported_by, 'Report submitted', reported_at FROM crime_reports
            WHERE Record_ID NOT IN (SELECT record_id FROM status_history)""")
        has_admin = con.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1").fetchone()
    if not has_admin:
        create_user(DEFAULT_ADMIN[0], DEFAULT_ADMIN[1], "admin", DEFAULT_ADMIN[2])


# ---------------------------------------------------------------- accounts
def create_user(username, password, role, full_name=""):
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not username.replace("_", "").replace(".", "").isalnum():
        return False, "Username may contain letters, numbers, dot and underscore only."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    salt = secrets.token_hex(16)
    try:
        with _conn() as con:
            con.execute("INSERT INTO users VALUES (?,?,?,?,?,?)",
                        (username, full_name.strip() or username, role, salt, _hash(password, salt), _now()))
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created."


def authenticate(username, password, role):
    """Returns the user dict if the credentials are valid for this role, else None."""
    with _conn() as con:
        row = con.execute("SELECT * FROM users WHERE username=?", (username.strip().lower(),)).fetchone()
    if row is None or row["role"] != role:
        return None
    if hmac.compare_digest(_hash(password, row["salt"]), row["password_hash"]):
        return {"username": row["username"], "name": row["full_name"], "role": row["role"]}
    return None


def list_users():
    with _conn() as con:
        return pd.read_sql_query(
            "SELECT username, full_name, role, created_at FROM users ORDER BY role, created_at", con)


# ---------------------------------------------------------------- login history
def log_event(username, role, event):
    with _conn() as con:
        con.execute("INSERT INTO login_history (username, role, event, timestamp) VALUES (?,?,?,?)",
                    (username.strip().lower(), role, event, _now()))


def get_history(role):
    with _conn() as con:
        return pd.read_sql_query(
            "SELECT id AS Entry, username AS Username, role AS Role, event AS Event, timestamp AS Timestamp "
            "FROM login_history WHERE role=? ORDER BY id DESC", con, params=(role,))


# ---------------------------------------------------------------- crime reports
def add_report(reported_by, record, notes=""):
    """Stores one user-registered crime and returns its Record_ID."""
    cols = [c for c in DATA_COLUMNS if c != "Record_ID"]
    with _conn() as con:
        cur = con.execute(
            f"INSERT INTO crime_reports (reported_by, reported_at, notes, status, {', '.join(cols)}) "
            f"VALUES (?,?,?,?,{', '.join('?' * len(cols))})",
            [reported_by, _now(), notes, STATUSES[0]] + [record[c] for c in cols])
        record_id = f"U{cur.lastrowid:05d}"
        con.execute("UPDATE crime_reports SET Record_ID=? WHERE id=?", (record_id, cur.lastrowid))
        con.execute("INSERT INTO status_history (record_id, status, changed_by, remarks, timestamp) VALUES (?,?,?,?,?)",
                    (record_id, STATUSES[0], reported_by, "Report submitted", _now()))
    return record_id


def get_reports():
    with _conn() as con:
        return pd.read_sql_query("SELECT * FROM crime_reports ORDER BY id DESC", con)


def delete_report(record_id):
    with _conn() as con:
        con.execute("DELETE FROM crime_reports WHERE Record_ID=?", (record_id,))
        con.execute("DELETE FROM status_history WHERE record_id=?", (record_id,))


def update_status(record_id, new_status, changed_by, remarks=""):
    if new_status not in STATUSES:
        return False
    with _conn() as con:
        con.execute("UPDATE crime_reports SET status=? WHERE Record_ID=?", (new_status, record_id))
        con.execute("INSERT INTO status_history (record_id, status, changed_by, remarks, timestamp) VALUES (?,?,?,?,?)",
                    (record_id, new_status, changed_by, remarks.strip(), _now()))
    return True


def get_status_history(record_id):
    with _conn() as con:
        return pd.read_sql_query(
            "SELECT timestamp AS Time, status AS Status, changed_by AS 'Updated By', remarks AS Remarks "
            "FROM status_history WHERE record_id=? ORDER BY id", con, params=(record_id,))
