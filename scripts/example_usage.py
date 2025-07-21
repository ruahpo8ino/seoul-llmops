#!/usr/bin/env python3
"""
서울시 LLM API 사용 예제
"""

import requests
import json
import time

# API 서버 설정
API_BASE_URL = "http://localhost:8000"

def test_health():
    """헬스 체크 테스트"""
    print("=== 헬스 체크 ===")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"상태 코드: {response.status_code}")
        print(f"응답: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except requests.exceptions.RequestException as e:
        print(f"오류 발생: {e}")
    print()

def test_generate():
    """텍스트 생성 테스트"""
    print("=== 텍스트 생성 테스트 ===")
    
    test_prompts = [
        "서울시의 인구는 얼마인가요?",
        "서울시청의 위치를 알려주세요",
        "서울의 대표적인 관광지는 어디인가요?",
        "지하철 이용방법을 알려주세요"
    ]
    
    for prompt in test_prompts:
        print(f"질문: {prompt}")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json={
                    "prompt": prompt,
                    "max_new_tokens": 100,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"답변: {result['generated_text']}")
                print(f"모델: {result['model_info']['model_path']}")
            else:
                print(f"오류 상태 코드: {response.status_code}")
                print(f"오류 메시지: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"요청 오류: {e}")
        
        print("-" * 50)
        time.sleep(1)  # 요청 간 대기

def test_chat():
    """채팅 인터페이스 테스트"""
    print("=== 채팅 인터페이스 테스트 ===")
    
    questions = [
        "안녕하세요, 서울시에 대해 알려주세요",
        "서울시의 면적은 얼마나 되나요?",
        "서울의 날씨는 어떤가요?"
    ]
    
    for question in questions:
        print(f"Q: {question}")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={"prompt": question}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"A: {result['response']}")
            else:
                print(f"오류: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"요청 오류: {e}")
        
        print()

def interactive_chat():
    """대화형 채팅"""
    print("=== 대화형 채팅 (종료하려면 'quit' 입력) ===")
    
    while True:
        try:
            user_input = input("\n질문: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("채팅을 종료합니다.")
                break
            
            if not user_input:
                continue
            
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={"prompt": user_input}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"답변: {result['response']}")
            else:
                print(f"오류 발생: {response.status_code} - {response.text}")
                
        except KeyboardInterrupt:
            print("\n채팅을 종료합니다.")
            break
        except requests.exceptions.RequestException as e:
            print(f"연결 오류: {e}")
            print("서버가 실행 중인지 확인해주세요.")
            break

def benchmark_performance():
    """성능 벤치마크"""
    print("=== 성능 벤치마크 ===")
    
    prompt = "서울시에 대해 간단히 소개해주세요"
    num_requests = 5
    
    times = []
    
    for i in range(num_requests):
        print(f"요청 {i+1}/{num_requests}...")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json={
                    "prompt": prompt,
                    "max_new_tokens": 50,
                    "temperature": 0.7
                }
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            times.append(response_time)
            
            if response.status_code == 200:
                print(f"응답 시간: {response_time:.2f}초")
            else:
                print(f"오류: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"요청 오류: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n성능 요약:")
        print(f"평균 응답 시간: {avg_time:.2f}초")
        print(f"최소 응답 시간: {min_time:.2f}초")
        print(f"최대 응답 시간: {max_time:.2f}초")

def main():
    """메인 함수"""
    print("서울시 LLM API 테스트 시작")
    print(f"API 서버: {API_BASE_URL}")
    print("=" * 60)
    
    # 1. 헬스 체크
    test_health()
    
    # 2. 텍스트 생성 테스트
    test_generate()
    
    # 3. 채팅 인터페이스 테스트
    test_chat()
    
    # 4. 성능 벤치마크
    benchmark_performance()
    
    # 5. 대화형 채팅 (선택사항)
    print("\n대화형 채팅을 시작하시겠습니까? (y/n): ", end="")
    if input().lower().startswith('y'):
        interactive_chat()

if __name__ == "__main__":
    main() 