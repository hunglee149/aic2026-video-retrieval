FROM python:3.12-slim

# Cài đặt các thư viện hệ thống cần thiết (nếu có compile C++ từ dependencies)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài đặt Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn dự án vào container
COPY . .

# Hugging Face Spaces mặc định chạy trên port 7860
EXPOSE 7860

# Khởi chạy server FastAPI bằng Uvicorn
CMD ["python", "-m", "uvicorn", "aic.ui.app:app", "--host", "0.0.0.0", "--port", "7860"]
