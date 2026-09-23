# ==========================================
# 🔐 설계사 로그인
# ==========================================
import sqlite3
import streamlit as st

DB_PATH = "agents.db"


def check_login(employee_no, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT employee_no FROM agents WHERE employee_no = ? AND password = ?",
        (employee_no, password)
    )

    result = cursor.fetchone()
    conn.close()

    return result is not None


