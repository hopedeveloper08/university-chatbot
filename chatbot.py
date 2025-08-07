import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv


############################################################################################
# Initialization
############################################################################################
st.markdown("""
<link href="https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font/dist/font-face.css" rel="stylesheet">
<style>
body [class^="st-"], body div {
    direction: rtl;
    font-family: 'Vazir', sans-serif !important;
    font-size: 1rem !important;
}
</style>
""", unsafe_allow_html=True)
st.write('### چت بات دانشگاه ولی عصر (عج)')


@st.cache_resource
def init_once():

    # LLM
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite', temperature=0)

    # Retriever
    embedding_model = HuggingFaceEmbeddings(model_name='embedding_model/')
    article_vector_store = FAISS.load_local('data/knowledge-base/article', embedding_model, allow_dangerous_deserialization=True)
    summary_vector_store = FAISS.load_local('data/knowledge-base/summary', embedding_model, allow_dangerous_deserialization=True)
    key_sentences_vector_store = FAISS.load_local('data/knowledge-base/key-sentences', embedding_model, allow_dangerous_deserialization=True)
    def retriever(query, task='article'):
        if task == 'article':
            return article_vector_store.similarity_search(query, k=3)
        elif task == 'summary':
            return summary_vector_store.similarity_search(query, k=3)
        elif task == 'key_sentences':
            return key_sentences_vector_store.similarity_search(query, k=3)

    # CoT  
    CoT_prompt = '''
تو یک مدل تخصصی برای تولید استدلال زنجیره‌ای (Chain of Thought - CoT) هستی. وظیفه تو این است که فقط بر اساس سوال کاربر و متن‌های بازیابی‌شده از آیین‌نامه‌های دانشگاه، یک استدلال زنجیره‌ای دقیق و مرحله‌به‌مرحله تولید کنی. نباید هیچ اطلاعات یا محتوای جدیدی از خودت اضافه کنی و تمام استدلال باید کاملاً مبتنی بر متن‌های بازیابی‌شده باشد. خروجی فقط باید شامل مراحل استدلال (CoT) باشد که به‌صورت واضح و ساختارمند نشان دهد چگونه از متن‌های بازیابی‌شده به پاسخ می‌رسی. پاسخ نهایی را تولید نکن، فقط CoT را ارائه بده.

**سوال کاربر:**
{question}

**متن‌های بازیابی‌شده:**
{context}

فرمت خروجی:
مرحله اول: [تحلیل سوال و ارتباط آن با متن بازیابی‌شده]
مرحله دوم: [استخراج اطلاعات مرتبط از متن و توضیح ارتباط]
مرحله سوم: [استدلال مرحله‌به‌مرحله برای رسیدن به پاسخ]
    '''.strip()
    CoT_prompt_template = PromptTemplate(input_variables=['context', 'question'], template=CoT_prompt)
    CoT = CoT_prompt_template | llm

    # Generative 
    generative_prompt = '''
تو یک مدل تولید پاسخ برای دانشگاه ولی عصر رفسنجان هستی. وظیفه تو این است که بر اساس سوال کاربر، متن‌های بازیابی‌شده از آیین‌نامه‌های دانشگاه، و استدلال زنجیره‌ای (CoT) ارائه‌شده، یک پاسخ دقیق، مختصر و مفید و خیلی خلاصه و همچنین کاملاً مبتنی بر متن‌های بازیابی‌شده تولید کنی. نباید هیچ اطلاعات یا محتوای جدیدی از خودت اضافه کنی. پاسخ باید مستقیم، کاربردی و مطابق با استدلال CoT باشد.

**سوال کاربر:**
{question}

**متن‌های بازیابی‌شده:**
{context}

**استدلال زنجیره ای:**
{cot}
    '''.strip()
    generative_prompt_template = PromptTemplate(input_variables=['context', 'question', 'cot'], template=generative_prompt)
    generative = generative_prompt_template | llm

    return retriever, CoT, generative
retriever, CoT, generative = init_once()


############################################################################################
# System
############################################################################################
def generate(query):
    context = ''
    for task in ['article', 'summary', 'key_sentences']:
        context += '\n\n'.join(doc.page_content for doc in retriever(query, task))
    cot = CoT.invoke({
        'question': query,
        'context': context
    }).content
    for chunk in generative.stream({
        'question': query,
        'context': context,
        'cot': cot
    }):
        yield chunk.content


############################################################################################
# UI
############################################################################################
if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input('سوال خود را بپرسید...')


if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk in generate(user_input):  
            full_response += chunk
            placeholder.markdown(full_response.replace("\n", " <br> ") + "▌", unsafe_allow_html=True)
        
        placeholder.markdown(full_response.replace("\n", " <br> "), unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_response.replace("\n", " <br> ")})
