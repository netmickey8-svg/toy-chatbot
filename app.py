"""
제안서 챗봇 - Streamlit UI
"""
from pathlib import Path
import faulthandler
import logging
import traceback

import streamlit as st

from src.ui_analysis import render_analysis_tab
from src.ui_chat import render_chat_tab
from src.ui_indexing import render_sidebar
from src.ui_state import initialize_session_state, load_rag_chain


LOG_PATH = Path(__file__).parent / "app_runtime.log"
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
try:
    _fh = open(LOG_PATH, "a", encoding="utf-8")
    faulthandler.enable(_fh)
except Exception:
    pass
logging.info("app.py imported")


st.set_page_config(
    page_title="제안서 챗봇",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .assistant-message {
        background-color: #F5F5F5;
    }
    .source-card {
        background-color: #FFF3E0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    """메인 앱"""
    logging.info("main() start")
    initialize_session_state()

    st.markdown('<h1 class="main-header">📄 제안서 챗봇</h1>', unsafe_allow_html=True)
    st.markdown("---")

    rag = load_rag_chain()
    logging.info("rag loaded; ready=%s", getattr(rag, "is_ready", lambda: False)())
    render_sidebar(rag)

    if not rag.is_ready():
        st.info(
            """
        현재 인덱스가 없습니다.

        서버는 그대로 사용할 수 있고, 사이드바 `인덱싱 관리`에서 `전체 인덱싱 실행` 또는
        업로드 후 수동 인덱싱을 실행하면 채팅이 활성화됩니다.
        """
        )
        return

    chat_tab, analysis_tab = st.tabs(["챗봇", "클러스터 분석"])
    with chat_tab:
        render_chat_tab(rag)
    with analysis_tab:
        render_analysis_tab(rag)


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is None:
            print("이 앱은 직접 실행이 아니라 아래 명령으로 실행해야 합니다:")
            print("python -m streamlit run app.py --server.fileWatcherType none")
            raise SystemExit(0)
    except ImportError:
        pass

    try:
        main()
    except Exception as error:
        logging.error("Unhandled exception: %s", error)
        logging.error(traceback.format_exc())
        raise
