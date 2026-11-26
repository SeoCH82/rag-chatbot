# 개인 RAG 챗봇

Streamlit에서 문서를 검색하고, 이 PC의 Ollama 모델로 답변하는 인공지능 과제입니다.

## Windows 실행

1. Ollama를 설치하고 과제에서 제공한 GGUF를 `model/professor_ljh_gguf/`에 넣습니다.
2. PowerShell에서 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1`로 Python 환경을 준비합니다.
3. 가상 시연 자료는 `RUN_LAN.cmd`로 실행합니다. 본인 자료로 실행하려면 기존 로컬 백업에서 `data/`와 `faiss_index/`를 복원하고 `RUN_PERSONAL.cmd`를 실행합니다.

ngrok 공유는 `deployment_config.json`의 `backend_url`에 본인 주소를 넣고 `SETUP_NGROK.cmd`로 인증한 다음 `RUN_NGROK.cmd`로 실행합니다.

## Streamlit Cloud 시연

`cloud-demo/app.py`를 진입점으로 사용합니다. 이 화면은 가상의 시연 자료를 검색하고, `RUN_PDF_LANGSERVE.cmd`로 여는 데스크톱 백엔드에 답변을 요청합니다. Streamlit Secrets에 `BACKEND_URL`과 `BACKEND_API_KEY`를 등록합니다.
