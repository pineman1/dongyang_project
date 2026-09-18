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


# 로그인 상태 초기화
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "employee_no" not in st.session_state:
    st.session_state.employee_no = None


# ==========================================
# 로그인하지 않은 경우
# ==========================================
if not st.session_state.logged_in:

    st.title("🔐 동양생명 AI FC 어시스턴트")
    st.subheader("보험 설계사 로그인")

    employee_no = st.text_input("사번", placeholder="예: A0001")
    password = st.text_input("비밀번호", type="password")

    if st.button("로그인", use_container_width=True):

        if check_login(employee_no, password):
            st.session_state.logged_in = True
            st.session_state.employee_no = employee_no
            st.rerun()

        else:
            st.error("사번 또는 비밀번호가 올바르지 않습니다.")

    st.stop()


# ==========================================
# 로그인 성공 후
# ==========================================
st.title("💼 동양생명 하이브리드 세일즈 어시스턴트")

st.write(
    f"로그인된 설계사: **{st.session_state.employee_no}**"
)

if st.button("로그아웃"):
    st.session_state.logged_in = False
    st.session_state.employee_no = None
    st.rerun()