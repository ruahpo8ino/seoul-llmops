# 서울시 LLMOps 파이프라인

서울시 AI Agent를 위한 오픈소스 LLM 학습 및 검증 파이프라인

## 📋 개요

이 프로젝트는 서울시를 위한 작은 LLM(Large Language Model) 모델 학습 파이프라인입니다. [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) 프레임워크를 활용하여 효율적인 모델 학습을 제공하고, FastAPI 기반의 추론 서버를 통해 API 서비스를 제공합니다.

### 🎯 주요 특징

- **Axolotl 기반 학습**: 최신 LoRA 파인튜닝 기술 활용
- **작은 모델 최적화**: 제한된 자원으로도 효과적인 학습
- **FastAPI 서버**: RESTful API를 통한 추론 서비스
- **Docker 지원**: 컨테이너 기반 배포 및 실행
- **서울시 특화**: 서울시 관련 데이터셋으로 사전 구성

## 🚀 빠른 시작

### 전제 조건

- Python 3.10+
- NVIDIA GPU (권장)
- Docker & Docker Compose
- 8GB+ GPU 메모리 (권장)

### 1. 프로젝트 클론

```bash
git clone https://github.com/your-repo/seoul-llmops.git
cd seoul-llmops
```

### 2. 환경 설정

#### 옵션 A: Docker 사용 (권장)

```bash
# 이미지 빌드
docker-compose build

# API 서버 실행
docker-compose --profile api up -d

# 학습 실행
docker-compose --profile training up
```

#### 옵션 B: 로컬 환경

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# accelerate 설정
accelerate config
```

### 3. 모델 학습

```bash
# 학습 실행
python scripts/train_model.py --config config/seoul_llm_config.yml

# 특정 단계만 실행
python scripts/train_model.py --skip-preprocess  # 전처리 건너뛰기
python scripts/train_model.py --skip-training    # 학습 건너뛰기
```

### 4. API 서버 실행

```bash
# 로컬 실행
python api/server.py --host 0.0.0.0 --port 8000

# 특정 모델 경로 지정
python api/server.py --model-path ./models/seoul-llm-lora/merged
```

### 5. API 테스트

```bash
# 헬스 체크
curl http://localhost:8000/health

# 텍스트 생성
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "서울시의 인구는 얼마인가요?"}'

# 예제 스크립트 실행
python scripts/example_usage.py
```

## 📁 프로젝트 구조

```
seoul-llmops/
├── api/                    # API 서버
│   └── server.py          # FastAPI 서버
├── config/                # 설정 파일
│   └── seoul_llm_config.yml
├── data/                  # 데이터셋
│   └── seoul_dataset.jsonl
├── scripts/               # 실행 스크립트
│   ├── train_model.py     # 학습 파이프라인
│   └── example_usage.py   # API 사용 예제
├── models/                # 모델 저장소
├── logs/                  # 로그 파일
├── Dockerfile             # Docker 설정
├── docker-compose.yml     # Docker Compose
├── requirements.txt       # Python 의존성
└── README.md             # 문서
```

## ⚙️ 설정

### 학습 설정 (config/seoul_llm_config.yml)

주요 설정 항목:

```yaml
# 베이스 모델
base_model: microsoft/DialoGPT-small

# LoRA 설정
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

# 학습 파라미터
micro_batch_size: 4
gradient_accumulation_steps: 2
num_epochs: 3
learning_rate: 0.0002

# 데이터셋
datasets:
  - path: data/seoul_dataset.jsonl
    ds_type: json
    type:
      system_prompt: "서울시 AI 어시스턴트로서..."
```

### 환경 변수

```bash
# 모델 경로
export MODEL_PATH="./models/seoul-llm-lora/merged"

# LoRA 어댑터 경로
export LORA_PATH="./models/seoul-llm-lora"

# GPU 설정
export CUDA_VISIBLE_DEVICES=0
```

## 🔧 사용법

### 1. 데이터셋 준비

JSONL 형식으로 데이터 준비:

```json
{"instruction": "질문", "input": "추가 입력(선택)", "output": "답변"}
```

### 2. 모델 학습

```bash
# 전체 파이프라인 실행
python scripts/train_model.py

# 단계별 실행
python scripts/train_model.py --skip-preprocess
python scripts/train_model.py --skip-training --skip-merge
```

### 3. API 사용

#### 텍스트 생성

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "서울시에 대해 알려주세요",
    "max_new_tokens": 100,
    "temperature": 0.7
})

print(response.json()["generated_text"])
```

#### 채팅 인터페이스

```python
response = requests.post("http://localhost:8000/chat", json={
    "prompt": "안녕하세요"
})

print(response.json()["response"])
```

## 🐳 Docker 사용법

### 프로필별 실행

```bash
# API 서버만 실행
docker-compose --profile api up -d

# 학습 환경 실행
docker-compose --profile training up

# 개발 환경 (Jupyter 포함)
docker-compose --profile dev up -d
```

### 볼륨 관리

```bash
# 모델 데이터 백업
docker run --rm -v seoul-llmops_huggingface_cache:/data alpine tar czf /backup.tar.gz /data

# 로그 확인
docker-compose logs seoul-llm-api
```

## 📊 성능 최적화

### GPU 메모리 최적화

```yaml
# config/seoul_llm_config.yml
fp16: true                    # 반정밀도 사용
gradient_checkpointing: true  # 메모리 절약
micro_batch_size: 2          # 배치 크기 감소
```

### 추론 최적화

```python
# API 서버에서 캐싱 활용
generation_config = GenerationConfig(
    use_cache=True,
    do_sample=True,
    temperature=0.7
)
```

## 🔍 모니터링

### 로그 확인

```bash
# 실시간 로그
tail -f logs/training.log

# Docker 로그
docker-compose logs -f seoul-llm-api
```

### Weights & Biases (선택사항)

```yaml
# config/seoul_llm_config.yml
wandb_mode: online
wandb_project: seoul-llm-training
wandb_entity: your-team
```

## 🛠️ 트러블슈팅

### 일반적인 문제

1. **CUDA 메모리 부족**
   ```yaml
   micro_batch_size: 1
   gradient_accumulation_steps: 4
   fp16: true
   ```

2. **모델 로드 실패**
   ```bash
   # 권한 확인
   chmod -R 755 models/
   
   # 디스크 공간 확인
   df -h
   ```

3. **API 서버 연결 실패**
   ```bash
   # 포트 확인
   netstat -tulpn | grep 8000
   
   # 방화벽 설정
   sudo ufw allow 8000
   ```

### 디버깅

```bash
# 상세 로그 활성화
export PYTHONPATH=$PWD
python -u scripts/train_model.py --config config/seoul_llm_config.yml

# API 디버그 모드
python api/server.py --reload
```

## 📈 확장 가능성

### 더 큰 모델 사용

```yaml
# config/seoul_llm_config.yml
base_model: microsoft/DialoGPT-medium  # 또는 large
lora_r: 32
lora_alpha: 64
```

### 다중 GPU 학습

```yaml
# DeepSpeed 설정
deepspeed: deepspeed/zero2.json
```

### 프로덕션 배포

```bash
# Gunicorn 사용
gunicorn api.server:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 Apache 2.0 라이선스 하에 배포됩니다.

## 📞 지원

- 이슈 리포트: [GitHub Issues](https://github.com/your-repo/seoul-llmops/issues)
- 문의: your-email@seoul.go.kr

---

## 📚 참고 자료

- [Axolotl Documentation](https://docs.axolotl.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Transformers Library](https://huggingface.co/docs/transformers/)
