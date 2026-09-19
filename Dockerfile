FROM python:3.11-slim
WORKDIR /app

# 클라우드 컨테이너 내 PDF 한글 깨짐 방지용 폰트 설치
RUN apt-get update && apt-get install -y fonts-nanum

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
