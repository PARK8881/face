# 🔮 AI 관상가

얼굴 사진을 업로드하면 OpenAI 이미지 분석 모델이 재미로 관상을 풀이해 주는 Streamlit 앱입니다.

> 본 결과는 재미로 보는 것이며 과학적 근거가 없습니다. 업로드한 사진은 저장되지 않습니다.

## 로컬 실행

```bash
# 1. 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 환경변수 설정: .env.example을 복사해 .env를 만들고 API 키 입력
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux

# 4. 실행
streamlit run app.py
```

`.env` 항목:

| 키 | 설명 | 기본값 |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API 키 (필수) | — |
| `OPENAI_MODEL` | 사용할 Vision 모델 | `gpt-4o-mini` |

## Streamlit Cloud 배포

1. `.env`가 커밋되지 않았는지 확인 후 GitHub에 push
2. [share.streamlit.io](https://share.streamlit.io)에서 저장소와 `app.py`를 선택해 배포
3. 앱 **Settings → Secrets**에 입력

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"
```
