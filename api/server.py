#!/usr/bin/env python3
"""
서울시 LLM API 서버
학습된 모델을 사용하여 텍스트 생성 API를 제공합니다.
"""

import os
import sys
import logging
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig
)
from peft import PeftModel

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 전역 변수
model = None
tokenizer = None
generation_config = None

class ModelManager:
    """모델 관리 클래스"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.model_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, model_path: str, lora_path: Optional[str] = None):
        """모델 로드"""
        try:
            logger.info(f"모델을 로드합니다: {model_path}")
            logger.info(f"사용 장치: {self.device}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # LoRA 어댑터 로드 (있는 경우)
            if lora_path and Path(lora_path).exists():
                logger.info(f"LoRA 어댑터를 로드합니다: {lora_path}")
                self.model = PeftModel.from_pretrained(
                    self.model, 
                    lora_path
                )
                # LoRA를 베이스 모델에 병합
                self.model = self.model.merge_and_unload()
            
            # CPU로 이동 (필요한 경우)
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # 생성 설정
            self.generation_config = GenerationConfig(
                max_length=512,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0,
            )
            
            logger.info("모델 로드가 완료되었습니다.")
            self.model_path = model_path
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {e}")
            raise
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        
        try:
            # 프롬프트 포맷팅
            formatted_prompt = f"### 질문: {prompt}\n### 답변:"
            
            # 토크나이징
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=400
            )
            
            # 장치로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 생성 설정 업데이트
            gen_config = GenerationConfig(**self.generation_config.to_dict())
            for key, value in kwargs.items():
                if hasattr(gen_config, key):
                    setattr(gen_config, key, value)
            
            # 텍스트 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    use_cache=True
                )
            
            # 결과 디코딩
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # 원본 프롬프트 제거하고 답변 부분만 추출
            answer_start = generated_text.find("### 답변:")
            if answer_start != -1:
                answer = generated_text[answer_start + len("### 답변:"):].strip()
            else:
                answer = generated_text[len(formatted_prompt):].strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"텍스트 생성 중 오류 발생: {e}")
            raise

# 전역 모델 매니저
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 라이프사이클 관리"""
    # 시작 시 모델 로드
    model_path = os.getenv("MODEL_PATH", "./models/seoul-llm-lora/merged")
    lora_path = os.getenv("LORA_PATH", None)
    
    # 경로가 존재하지 않으면 대체 경로 시도
    if not Path(model_path).exists():
        alternative_paths = [
            "./models/seoul-llm",
            "./models/seoul-llm-lora",
            "microsoft/DialoGPT-small"  # 기본 모델
        ]
        
        for alt_path in alternative_paths:
            if Path(alt_path).exists() or alt_path.startswith("microsoft/"):
                model_path = alt_path
                logger.info(f"대체 모델 경로를 사용합니다: {model_path}")
                break
    
    try:
        model_manager.load_model(model_path, lora_path)
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        logger.info("기본 모델을 로드합니다...")
        model_manager.load_model("microsoft/DialoGPT-small")
    
    yield
    
    # 종료 시 정리
    logger.info("서버를 종료합니다...")

# FastAPI 앱 생성
app = FastAPI(
    title="서울시 LLM API",
    description="서울시 AI 어시스턴트를 위한 LLM API 서버",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델
class GenerateRequest(BaseModel):
    """텍스트 생성 요청"""
    prompt: str = Field(..., description="생성할 텍스트의 프롬프트")
    max_new_tokens: Optional[int] = Field(150, description="최대 생성 토큰 수")
    temperature: Optional[float] = Field(0.7, description="생성 다양성 (0.0-2.0)")
    top_p: Optional[float] = Field(0.9, description="누적 확률 임계값")
    top_k: Optional[int] = Field(50, description="상위 K개 토큰 선택")
    repetition_penalty: Optional[float] = Field(1.1, description="반복 억제 정도")

class GenerateResponse(BaseModel):
    """텍스트 생성 응답"""
    generated_text: str = Field(..., description="생성된 텍스트")
    prompt: str = Field(..., description="입력 프롬프트")
    model_info: Dict[str, Any] = Field(..., description="모델 정보")

class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    model_loaded: bool
    model_path: Optional[str]
    device: str

# API 엔드포인트
@app.get("/", response_model=Dict[str, str])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "서울시 LLM API 서버",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크"""
    return HealthResponse(
        status="healthy" if model_manager.model is not None else "unhealthy",
        model_loaded=model_manager.model is not None,
        model_path=model_manager.model_path,
        device=model_manager.device
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """텍스트 생성"""
    if model_manager.model is None:
        raise HTTPException(
            status_code=503,
            detail="모델이 로드되지 않았습니다."
        )
    
    try:
        # 생성 파라미터 준비
        generation_params = {
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
        }
        
        # 텍스트 생성
        generated_text = model_manager.generate_text(
            request.prompt,
            **generation_params
        )
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            model_info={
                "model_path": model_manager.model_path,
                "device": model_manager.device,
                "generation_params": generation_params
            }
        )
        
    except Exception as e:
        logger.error(f"텍스트 생성 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"텍스트 생성 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/chat")
async def chat(request: GenerateRequest):
    """간단한 채팅 인터페이스"""
    if model_manager.model is None:
        raise HTTPException(
            status_code=503,
            detail="모델이 로드되지 않았습니다."
        )
    
    try:
        generated_text = model_manager.generate_text(request.prompt)
        return {"response": generated_text}
        
    except Exception as e:
        logger.error(f"채팅 응답 생성 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"응답 생성 중 오류가 발생했습니다: {str(e)}"
        )

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="서울시 LLM API 서버")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument("--model-path", help="모델 경로")
    parser.add_argument("--lora-path", help="LoRA 어댑터 경로")
    parser.add_argument("--reload", action="store_true", help="개발 모드 (자동 재로드)")
    
    args = parser.parse_args()
    
    # 환경 변수 설정
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    if args.lora_path:
        os.environ["LORA_PATH"] = args.lora_path
    
    logger.info(f"서버를 시작합니다: http://{args.host}:{args.port}")
    logger.info(f"API 문서: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main() 