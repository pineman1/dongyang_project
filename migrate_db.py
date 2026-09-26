import hashlib
import sqlite3

DB_PATH = "agents.db"


def migrate_passwords():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # '1234'를 SHA-256으로 해싱한 64자리 16진수 문자열
    # (03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4)
    hashed_1234 = hashlib.sha256("1234".encode("utf-8")).hexdigest()

    # 기존 '1234'로 저장된 행을 찾아 해시값으로 변경
    cursor.execute(
        "UPDATE agents SET password = ? WHERE password = '1234'",
        (hashed_1234,),
    )

    conn.commit()
    print(f"업데이트된 행 개수: {cursor.rowcount}")
    conn.close()
    print("DB 암호화 마이그레이션이 성공적으로 완료되었습니다!")


if __name__ == "__main__":
    migrate_passwords()