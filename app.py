"""AI 관상가 — 얼굴 사진을 업로드하면 OpenAI Vision 모델이 관상을 풀이해 주는 Streamlit 앱."""

import base64
import hashlib
import io
import os
import time

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError

load_dotenv()  # 로컬 실행 시 .env 로드 (Streamlit Cloud에서는 st.secrets 사용)

ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_IMAGE_SIDE = 1024
MAX_IMAGE_PIXELS = 50_000_000  # 약 50MP: 4800만 화소 휴대폰 사진까지 허용
DEFAULT_MODEL = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 1500
API_TIMEOUT_SECONDS = 60

NO_FACE = "NO_FACE"
MULTI_FACE = "MULTI_FACE"
DISCLAIMER = "본 결과는 재미로 보는 것이며 과학적 근거가 없습니다."

SYSTEM_PROMPT = f"""당신은 수십 년 경력의 따뜻하고 유쾌한 관상가입니다.
사용자가 보낸 얼굴 사진을 전통 관상학의 관점에서 재미로 풀이해 주세요.

규칙:
- 반드시 한국어로, 긍정적이고 따뜻한 톤으로 작성합니다. 단점도 장점이나 조언으로 바꿔 말합니다.
- 인종, 성별, 나이, 외모의 우열을 평가하거나 비하하지 않고, 단정적으로 운명을 예언하지 않습니다.
- 사람을 식별하거나 이름을 추측하지 않습니다.
- 사진에 사람 얼굴이 없거나 얼굴을 알아볼 수 없으면 다른 말 없이 정확히 `{NO_FACE}`만 출력합니다.
- 사진에 얼굴이 2명 이상이면 다른 말 없이 정확히 `{MULTI_FACE}`만 출력합니다.

얼굴이 한 명일 때는 아래 마크다운 형식을 그대로 따릅니다.

## 첫인상
(한두 문장 요약)

## 부위별 풀이
- **이마**: ...
- **눈썹**: ...
- **눈**: ...
- **코**: ...
- **입**: ...
- **귀**: ... (사진에서 보이지 않으면 보이지 않는다고 짧게 언급)
- **턱/얼굴형**: ...

## 운세 풀이
- **재물운**: ...
- **애정운**: ...
- **직업운**: ...
- **건강운**: ...

## 총평과 한마디
(전체 요약과 따뜻한 조언 한마디)
"""

MOCK_RESULT = """## 첫인상
밝고 단단한 기운이 느껴지는 얼굴입니다. 처음 만난 사람도 편안하게 만드는 인상이에요.

## 부위별 풀이
- **이마**: 넓고 시원한 이마는 초년운이 좋고 생각이 깊음을 뜻합니다.
- **눈썹**: 가지런한 눈썹은 형제·친구와의 인연이 두텁다는 의미예요.
- **눈**: 맑은 눈빛은 판단력과 집중력이 뛰어남을 보여줍니다.
- **코**: 콧대가 곧아 재물을 모으고 지키는 힘이 있습니다.
- **입**: 입꼬리가 부드러워 말로 복을 부르는 상입니다.
- **귀**: 사진에서 잘 보이지 않지만, 전체 균형으로 보아 복이 있는 편입니다.
- **턱/얼굴형**: 안정적인 턱선은 말년운이 편안함을 의미합니다.

## 운세 풀이
- **재물운**: 꾸준히 모으는 힘이 있어 차곡차곡 쌓이는 재물운입니다.
- **애정운**: 진심이 통하는 인연을 만나기 좋은 시기입니다.
- **직업운**: 신뢰를 바탕으로 한 분야에서 인정받는 운입니다.
- **건강운**: 기본 체력이 좋으니 규칙적인 생활만 지키면 충분합니다.

## 총평과 한마디
복이 고르게 깃든 균형 잡힌 관상입니다. 지금처럼 웃는 얼굴을 유지하면 좋은 일이 더 찾아올 거예요!
"""


class AnalysisError(Exception):
    """사용자에게 그대로 보여줄 수 있는 분석 오류."""


def get_secret(name: str, default: str | None = None) -> str | None:
    """st.secrets(Streamlit Cloud) → 환경변수(.env) 순서로 값을 찾는다."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # secrets.toml이 없으면 st.secrets 접근 시 예외가 발생한다.
        pass
    return os.getenv(name, default)


# ---------------------------------------------------------------------------
# 이미지 처리 (메모리에서만 처리, 디스크에 저장하지 않음)
# ---------------------------------------------------------------------------
def load_image(data: bytes) -> Image.Image:
    """바이트를 PIL 이미지로 열고 EXIF 회전 보정 및 RGB 변환을 한다.

    파일은 작아도 픽셀 수가 매우 큰 이미지(압축 폭탄)로 메모리가 고갈되지 않도록,
    헤더만 읽은 상태에서 픽셀 수를 먼저 검사한 뒤 디코딩한다.
    """
    too_large = AnalysisError(
        f"사진 해상도가 너무 커요. {MAX_IMAGE_PIXELS // 1_000_000:,}메가픽셀 이하의 사진을 올려주세요."
    )
    try:
        image = Image.open(io.BytesIO(data))  # 헤더만 읽음 (아직 디코딩 전)
    except Image.DecompressionBombError as exc:
        raise too_large from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise AnalysisError("이미지를 열 수 없어요. 손상되지 않은 JPG/PNG/WEBP 파일인지 확인해 주세요.") from exc

    width, height = image.size
    if width * height > MAX_IMAGE_PIXELS:
        raise too_large

    try:
        # JPEG는 필요한 크기에 가깝게 축소하며 디코딩해 메모리를 아낀다 (다른 형식은 무시됨).
        image.draft("RGB", (MAX_IMAGE_SIDE * 2, MAX_IMAGE_SIDE * 2))
        image.load()
    except Image.DecompressionBombError as exc:
        raise too_large from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise AnalysisError("이미지를 열 수 없어요. 손상되지 않은 JPG/PNG/WEBP 파일인지 확인해 주세요.") from exc
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def resize_image(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """긴 변이 max_side 이하가 되도록 비율을 유지하며 축소한다."""
    image = image.copy()
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def encode_image(image: Image.Image) -> str:
    """이미지를 JPEG base64 data URL로 인코딩한다."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# 관상 분석
# ---------------------------------------------------------------------------
def classify_response(text: str) -> str:
    """모델 응답이 NO_FACE / MULTI_FACE 신호인지 판별한다.

    모델이 신호를 문장과 섞어 답하는 경우도 있으므로, 관상 형식의 제목(##)이 없고
    신호 문자열이 포함되어 있으면 신호로 본다.
    """
    upper = text.upper()
    if "##" not in text:
        if MULTI_FACE in upper:
            return MULTI_FACE
        if NO_FACE in upper:
            return NO_FACE
    return "OK"


def analyze_face(image_data_url: str, api_key: str | None, model: str, mock: str | None = None) -> str:
    """OpenAI Vision 모델로 관상을 분석해 원문 응답을 반환한다."""
    if mock:
        time.sleep(1)
        if mock.lower() == "no_face":
            return NO_FACE
        if mock.lower() == "multi_face":
            return MULTI_FACE
        return MOCK_RESULT

    client = OpenAI(api_key=api_key, timeout=API_TIMEOUT_SECONDS, max_retries=1)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "이 사람의 관상을 봐주세요."},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
    except openai.AuthenticationError as exc:
        raise AnalysisError("API 키가 올바르지 않아요. OPENAI_API_KEY 설정을 확인해 주세요.") from exc
    except openai.RateLimitError as exc:
        raise AnalysisError("요청이 많거나 사용 한도를 초과했어요. 잠시 후 다시 시도해 주세요.") from exc
    except (openai.APITimeoutError, openai.APIConnectionError) as exc:
        raise AnalysisError("OpenAI 서버에 연결하지 못했어요. 네트워크 상태를 확인하고 다시 시도해 주세요.") from exc
    except openai.NotFoundError as exc:
        raise AnalysisError(f"모델 '{model}'을(를) 사용할 수 없어요. OPENAI_MODEL 설정을 확인해 주세요.") from exc
    except openai.APIError as exc:
        raise AnalysisError("OpenAI 요청 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요.") from exc

    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise AnalysisError(f"모델이 이 사진의 분석을 거절했어요: {message.refusal}")
    if not message.content:
        raise AnalysisError("분석 결과가 비어 있어요. 다른 사진으로 다시 시도해 주세요.")
    return message.content.strip()




# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
SECTION_EMOJI = {
    "첫인상": "👀",
    "부위별 풀이": "🧭",
    "운세 풀이": "🍀",
    "총평과 한마디": "📜",
}

CUSTOM_CSS = """
<style>
.hero { text-align: center; padding: 0.5rem 0 1rem; }
.hero .icon { font-size: 3.2rem; line-height: 1; }
.hero h1 { font-size: 2.2rem; margin: 0.3rem 0 0.2rem; padding: 0; }
.hero p { opacity: 0.75; margin: 0; font-size: 1.05rem; }
.hero .rule {
    width: 64px; height: 3px; margin: 0.9rem auto 0;
    background: linear-gradient(90deg, #C8A04A, #8E3B5E); border-radius: 2px;
}
/* 결과 카드 안의 섹션 제목 크기 */
.st-key-result_card h2 { font-size: 1.35rem; margin-top: 0.6rem; }
/* 모바일: 가로 스크롤 방지 */
[data-testid="stImage"] img { max-width: 100%; height: auto; }
@media (max-width: 480px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .hero h1 { font-size: 1.7rem; }
}
</style>
"""


def decorate_headings(markdown_text: str) -> str:
    """결과의 섹션 제목(## ...) 앞에 이모지를 붙인다."""
    lines = []
    for line in markdown_text.splitlines():
        if line.startswith("## "):
            title = line[3:].strip()
            emoji = SECTION_EMOJI.get(title)
            if emoji:
                line = f"## {emoji} {title}"
        lines.append(line)
    return "\n".join(lines)


def reset_app() -> None:
    """결과와 입력을 비우고 초기 상태로 되돌린다."""
    st.session_state.input_version += 1  # 위젯 key를 바꿔 업로드/촬영 입력을 비움
    st.session_state.image_id = None
    st.session_state.result = None
    st.session_state.active_source = None


def set_active_source(name: str) -> None:
    """업로드/촬영 중 마지막으로 바뀐 입력을 기록한다."""
    st.session_state.active_source = name


st.set_page_config(page_title="AI 관상가", page_icon="🔮", layout="centered")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

mock_mode = os.getenv("MOCK_OPENAI") or None
if mock_mode == "0":
    mock_mode = None
api_key = get_secret("OPENAI_API_KEY")
model = get_secret("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL

with st.sidebar:
    st.markdown("### 🔮 AI 관상가")
    st.write("얼굴 사진 한 장으로 전통 관상학에 기반한 풀이를 재미로 들려드려요.")
    st.markdown("#### 사용 방법")
    st.markdown("1. 사진을 올리거나 카메라로 찍어요\n2. **관상 보기**를 눌러요\n3. 부위별 풀이와 운세를 확인해요")
    st.markdown("#### 개인정보 안내")
    st.caption("사진은 분석에만 쓰이고 서버에 저장되지 않아요. 분석을 위해 OpenAI로 전송됩니다.")
    st.divider()
    st.caption(f"사용 모델: `{'MOCK' if mock_mode else model}`")

st.markdown(
    """
    <div class="hero">
      <div class="icon">🔮</div>
      <h1>AI 관상가</h1>
      <p>사진 한 장으로 보는 나의 관상</p>
      <div class="rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info("재미로 보는 서비스예요 · 업로드한 사진은 저장되지 않아요", icon="ℹ️")

if mock_mode:
    st.warning("MOCK 모드입니다. OpenAI를 호출하지 않고 샘플 결과를 보여줍니다.")
elif not api_key:
    st.error(
        "OPENAI_API_KEY가 설정되지 않았습니다. "
        "로컬에서는 `.env` 파일에, Streamlit Cloud에서는 앱 설정의 Secrets에 키를 등록해 주세요."
    )
    st.stop()

st.session_state.setdefault("image_id", None)
st.session_state.setdefault("result", None)  # {"kind": "OK"|NO_FACE|MULTI_FACE, "text": str}
st.session_state.setdefault("input_version", 0)
st.session_state.setdefault("active_source", None)  # "upload" | "camera" | None
version = st.session_state.input_version

tab_upload, tab_camera = st.tabs(["📁 사진 업로드", "📷 카메라 촬영"])
with tab_upload:
    uploaded_file = st.file_uploader(
        "얼굴이 잘 보이는 정면 사진을 올려주세요",
        type=ALLOWED_TYPES,
        help=f"JPG · PNG · WEBP, 최대 {MAX_FILE_SIZE_MB}MB",
        key=f"uploader_{version}",
        on_change=set_active_source,
        args=("upload",),
    )
    # 10MB 초과 파일은 업로더가 아이콘만으로 거부하므로 제한을 눈에 보이게 안내한다.
    st.caption(f"JPG · PNG · WEBP · 최대 {MAX_FILE_SIZE_MB}MB")
with tab_camera:
    camera_file = st.camera_input(
        "카메라로 얼굴을 찍어주세요",
        key=f"camera_{version}",
        on_change=set_active_source,
        args=("camera",),
    )

# 마지막으로 바뀐 입력을 우선 사용하고, 그 입력이 비어 있으면 다른 입력으로 대체한다.
if st.session_state.active_source == "camera":
    source = camera_file or uploaded_file
else:
    source = uploaded_file or camera_file
image = None
error_message = None

if source is not None:
    data = source.getvalue()
    image_id = hashlib.sha256(data).hexdigest()
    if image_id != st.session_state.image_id:
        # 새 이미지가 들어오면 이전 결과 초기화
        st.session_state.image_id = image_id
        st.session_state.result = None

    if len(data) > MAX_FILE_SIZE_BYTES:
        error_message = f"파일이 너무 커요. {MAX_FILE_SIZE_MB}MB 이하의 사진을 올려주세요."
    else:
        try:
            image = resize_image(load_image(data))
        except AnalysisError as exc:
            error_message = str(exc)
        except Exception:
            # 예상하지 못한 이미지 처리 오류도 내부 정보 없이 안내한다.
            error_message = "사진을 처리하지 못했어요. 다른 사진으로 다시 시도해 주세요."
else:
    st.session_state.image_id = None
    st.session_state.result = None

if error_message:
    st.warning(error_message)
elif image is not None:
    with st.container(horizontal_alignment="center"):
        st.image(image, caption="분석할 사진", width=320)

if st.button("🔮 관상 보기", type="primary", disabled=image is None, width="stretch"):
    with st.spinner("관상을 살펴보는 중이에요… 🔮"):
        try:
            text = analyze_face(encode_image(image), api_key, model, mock=mock_mode)
            st.session_state.result = {"kind": classify_response(text), "text": text}
        except AnalysisError as exc:
            st.session_state.result = None
            st.error(str(exc))
        except Exception:
            st.session_state.result = None
            st.error("알 수 없는 오류가 발생했어요. 잠시 후 다시 시도해 주세요.")

result = st.session_state.result
if result:
    if result["kind"] == NO_FACE:
        st.warning("사진에서 얼굴을 찾지 못했어요. 얼굴이 잘 보이는 정면 사진으로 다시 시도해 주세요.")
    elif result["kind"] == MULTI_FACE:
        st.warning("사진에 여러 명이 있어요. 한 사람만 나온 사진으로 다시 시도해 주세요.")
    else:
        st.markdown("### 📜 관상 결과")
        with st.container(border=True, key="result_card"):
            st.markdown(decorate_headings(result["text"]))
        st.caption(f"※ {DISCLAIMER}")
    st.button("🔄 다른 사진으로 다시 보기", on_click=reset_app, width="stretch")
