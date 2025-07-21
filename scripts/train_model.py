#!/usr/bin/env python3
"""
서울시 LLM 모델 학습 파이프라인
Axolotl을 사용하여 작은 LLM 모델을 학습합니다.
"""

import os
import sys
import argparse
import subprocess
import yaml
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SeoulLLMTrainer:
    def __init__(self, config_path: str):
        """
        서울시 LLM 트레이너 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.project_root = Path(__file__).parent.parent
        
    def _load_config(self) -> dict:
        """설정 파일 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def setup_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.project_root / "models",
            self.project_root / "data" / "prepared",
            self.project_root / "logs",
            Path(self.config.get("output_dir", "./models/seoul-llm")),
            Path(self.config.get("lora_out_dir", "./models/seoul-llm-lora")),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"디렉토리 생성됨: {directory}")
    
    def validate_dataset(self):
        """데이터셋 존재 확인"""
        dataset_path = None
        if "datasets" in self.config and len(self.config["datasets"]) > 0:
            dataset_path = Path(self.config["datasets"][0]["path"])
        
        if not dataset_path or not dataset_path.exists():
            logger.warning(f"데이터셋 파일이 없습니다: {dataset_path}")
            logger.info("샘플 데이터셋을 생성합니다...")
            self._create_sample_dataset(dataset_path)
    
    def _create_sample_dataset(self, dataset_path: Path):
        """샘플 데이터셋 생성"""
        sample_data = [
            {
                "instruction": "서울시의 인구는 얼마인가요?",
                "input": "",
                "output": "서울특별시의 인구는 약 950만 명입니다. 정확한 수치는 통계청의 최신 인구 조사를 참고해 주세요."
            },
            {
                "instruction": "서울시청의 위치를 알려주세요",
                "input": "",
                "output": "서울시청은 서울특별시 중구 세종대로 110에 위치하고 있습니다. 지하철 1호선, 2호선 시청역에서 가까운 거리에 있습니다."
            },
            {
                "instruction": "서울시의 대표적인 관광지는 어디인가요?",
                "input": "",
                "output": "서울시의 대표적인 관광지로는 경복궁, 창덕궁 등의 고궁, 명동과 강남 등의 상업지구, 한강공원, N서울타워, 홍대와 이태원 등이 있습니다."
            },
            {
                "instruction": "서울시 대중교통 이용방법을 알려주세요",
                "input": "교통카드",
                "output": "서울시에서는 티머니카드나 원패스카드 등의 교통카드를 사용하여 지하철, 버스를 편리하게 이용할 수 있습니다. 모바일 앱을 통해서도 결제가 가능합니다."
            }
        ]
        
        # 데이터셋 디렉토리 생성
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSONL 형식으로 저장
        import json
        with open(dataset_path, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"샘플 데이터셋이 생성되었습니다: {dataset_path}")
    
    def preprocess_data(self):
        """데이터 전처리"""
        logger.info("데이터 전처리를 시작합니다...")
        
        cmd = [
            "python", "-m", "axolotl.cli.preprocess",
            str(self.config_path)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("데이터 전처리가 완료되었습니다.")
            logger.debug(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"데이터 전처리 중 오류 발생: {e}")
            logger.error(f"에러 출력: {e.stderr}")
            return False
        
        return True
    
    def train_model(self):
        """모델 학습"""
        logger.info("모델 학습을 시작합니다...")
        
        cmd = [
            "accelerate", "launch",
            "-m", "axolotl.cli.train",
            str(self.config_path)
        ]
        
        try:
            # 실시간 로그 출력을 위해 Popen 사용
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 실시간 로그 출력
            for line in process.stdout:
                print(line.rstrip())
                logger.info(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                logger.info("모델 학습이 완료되었습니다.")
                return True
            else:
                logger.error(f"모델 학습 중 오류 발생 (exit code: {process.returncode})")
                return False
                
        except Exception as e:
            logger.error(f"모델 학습 중 예외 발생: {e}")
            return False
    
    def merge_lora(self):
        """LoRA 어댑터를 베이스 모델과 병합"""
        if self.config.get("adapter") != "lora":
            logger.info("LoRA가 사용되지 않았으므로 병합을 건너뜁니다.")
            return True
        
        logger.info("LoRA 어댑터를 베이스 모델과 병합합니다...")
        
        lora_dir = Path(self.config.get("lora_out_dir", "./models/seoul-llm-lora"))
        if not lora_dir.exists():
            logger.error(f"LoRA 출력 디렉토리를 찾을 수 없습니다: {lora_dir}")
            return False
        
        cmd = [
            "python", "-m", "axolotl.cli.merge_lora",
            str(self.config_path),
            f"--lora_model_dir={lora_dir}",
            "--load_in_8bit=False",
            "--load_in_4bit=False"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("LoRA 병합이 완료되었습니다.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"LoRA 병합 중 오류 발생: {e}")
            logger.error(f"에러 출력: {e.stderr}")
            return False
    
    def run_pipeline(self, skip_preprocess=False, skip_training=False, skip_merge=False):
        """전체 학습 파이프라인 실행"""
        logger.info("=== 서울시 LLM 학습 파이프라인 시작 ===")
        
        # 1. 디렉토리 설정
        self.setup_directories()
        
        # 2. 데이터셋 검증
        self.validate_dataset()
        
        # 3. 데이터 전처리
        if not skip_preprocess:
            if not self.preprocess_data():
                logger.error("데이터 전처리 실패")
                return False
        
        # 4. 모델 학습
        if not skip_training:
            if not self.train_model():
                logger.error("모델 학습 실패")
                return False
        
        # 5. LoRA 병합
        if not skip_merge:
            if not self.merge_lora():
                logger.error("LoRA 병합 실패")
                return False
        
        logger.info("=== 서울시 LLM 학습 파이프라인 완료 ===")
        return True

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="서울시 LLM 모델 학습 파이프라인")
    parser.add_argument(
        "--config", 
        default="config/seoul_llm_config.yml",
        help="설정 파일 경로 (기본값: config/seoul_llm_config.yml)"
    )
    parser.add_argument(
        "--skip-preprocess", 
        action="store_true",
        help="데이터 전처리 건너뛰기"
    )
    parser.add_argument(
        "--skip-training", 
        action="store_true",
        help="모델 학습 건너뛰기"
    )
    parser.add_argument(
        "--skip-merge", 
        action="store_true",
        help="LoRA 병합 건너뛰기"
    )
    
    args = parser.parse_args()
    
    try:
        trainer = SeoulLLMTrainer(args.config)
        success = trainer.run_pipeline(
            skip_preprocess=args.skip_preprocess,
            skip_training=args.skip_training,
            skip_merge=args.skip_merge
        )
        
        if success:
            logger.info("학습 파이프라인이 성공적으로 완료되었습니다!")
            sys.exit(0)
        else:
            logger.error("학습 파이프라인이 실패했습니다.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 