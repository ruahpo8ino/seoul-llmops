# 서울시 LLM 학습 및 추론 환경을 위한 Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=ko_KR.UTF-8
ENV LANGUAGE=ko_KR:ko
ENV LC_ALL=ko_KR.UTF-8

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    locales \
    && rm -rf /var/lib/apt/lists/*

# 한국어 로케일 설정
RUN locale-gen ko_KR.UTF-8

# Python 3.10을 기본 python으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# pip 업그레이드
RUN pip install --upgrade pip setuptools wheel

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (axolotl 및 flash-attention을 위해 필요)
RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 필수 라이브러리 먼저 설치
COPY requirements.txt .
RUN pip install packaging setuptools wheel ninja

# PyTorch 설치 (CUDA 11.8 버전)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Flash Attention 설치 (소스 컴파일이 필요할 수 있음)
RUN pip install flash-attn --no-build-isolation

# 나머지 의존성 설치
RUN pip install -r requirements.txt

# 프로젝트 파일 복사
COPY . .

# 실행 권한 설정
RUN chmod +x scripts/train_model.py
RUN chmod +x api/server.py

# 필요한 디렉토리 생성
RUN mkdir -p models data logs config api scripts

# 포트 노출
EXPOSE 8000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 기본 명령어 (API 서버 실행)
CMD ["python", "api/server.py", "--host", "0.0.0.0", "--port", "8000"] 