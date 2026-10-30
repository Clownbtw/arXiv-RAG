"""Streamlit-интерфейс: вопрос -> ответ с источниками. Запуск: streamlit run app.py"""
import streamlit as st

from rag import RAG

st.set_page_config(page_title="arXiv RAG", page_icon="📚")
st.title("📚 Поиск по статьям arXiv")
st.caption("Аннотации cs.CL / cs.LG · E5 + Chroma + cross-encoder + LLM")


@st.cache_resource(show_spinner="Загружаю модели и строю индекс (несколько минут при первом запуске)...")
def get_rag() -> RAG:
    return RAG()


rag = get_rag()
MAX_QUESTIONS = 20          # защита лимитов LLM API на публичном демо
st.session_state.setdefault("history", [])
st.session_state.setdefault("count", 0)


def render(msg: dict) -> None:
    st.markdown(msg["answer"])
    if msg.get("sources"):
        with st.expander(f"Источники ({len(msg['sources'])})"):
            for s in msg["sources"]:
                st.markdown(
                    f"**[{s['n']}] {s['title']}**  \n"
                    f"`{s['categories']}` · rerank {s['rerank']:.2f} · [поиск на arXiv]({s['url']})")
                st.caption(s["text"][:400] + "...")


for role, msg in st.session_state.history:
    with st.chat_message(role):
        render(msg) if role == "assistant" else st.markdown(msg)

if q := st.chat_input("Например: какие есть методы сжатия нейросетей?"):
    st.session_state.history.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        if st.session_state.count >= MAX_QUESTIONS:
            msg = {"answer": "Лимит вопросов в этой сессии исчерпан."}
        else:
            try:
                with st.spinner("Ищу и формирую ответ..."):
                    msg = rag.answer(q)
                st.session_state.count += 1
            except Exception as e:
                msg = {"answer": f"Ошибка при обращении к LLM ({type(e).__name__}). Попробуйте позже."}
        render(msg)
    st.session_state.history.append(("assistant", msg))
