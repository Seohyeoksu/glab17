import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Streamlit 페이지 설정
st.set_page_config(page_title="교육 컨설턴트 챗봇", page_icon="🎓", layout="centered")

# 제목
st.title("🎓 교육 컨설턴트 챗봇")

# OpenAI API 키를 Streamlit secrets에서 가져옵니다.
openai_api_key = st.secrets["OPENAI_API_KEY"]

# 세션 상태 초기화
if "conversation" not in st.session_state:
    try:
        llm = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
        
        template = """당신은 교사들에게 학생들의 기초학력 향상을 위한 방법을 제안하는 전문 교육 컨설턴트입니다.
        교육 정책, 교수법, 학습 심리학에 대한 깊은 이해를 바탕으로 실용적이고 효과적인 조언을 제공합니다.

        현재 대화:
        {history}
        인간: {input}
        AI 조언자: """

        prompt = PromptTemplate(input_variables=["history", "input"], template=template)
        
        st.session_state.conversation = ConversationChain(
            llm=llm, 
            verbose=True, 
            memory=ConversationBufferMemory(human_prefix="인간", ai_prefix="AI 조언자"),
            prompt=prompt
        )
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        st.stop()

# 대화 기록 표시
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
user_input = st.chat_input("질문을 입력해주세요:")

# 사용자 입력 처리
if user_input:
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        # AI 응답 생성
        response = st.session_state.conversation.predict(input=user_input)

        # AI 응답 표시
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"응답 생성 중 오류가 발생했습니다: {str(e)}")

# 스타일 적용
st.markdown("""
<style>
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 푸터
st.markdown("""
<div style='text-align: center; color: #888; padding: 10px;'>
    <p>© 2024 교육 컨설턴트 챗봇. 모든 권리 보유.</p>
</div>
""", unsafe_allow_html=True)