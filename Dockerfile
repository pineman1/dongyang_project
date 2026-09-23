FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 및 한글 폰트 설치 (FPDF2 깨짐 방지)
RUN apt-get update && apt-get install -y \
    fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 필수 에셋 및 코드 복사 (누락 주의)
COPY dongyang_products.pdf .
COPY *.pkl ./
COPY pdf_generator.py .
COPY app.py .
# 추가로 필요한 커스텀 모듈이나 데이터 폴더가 있다면 이곳에 COPY 추가

# 외부 접속을 위한 포트 노출
EXPOSE 8501

# 컨테이너 실행 시 엔트리포인트
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
