import streamlit as st
import base64
import requests
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
model_name = "gpt-4o"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())

    return file_path

def extract_text(image_path):
    query = '다음 그림은 SAP에서 에러메시지 표출 화면입니다. 화면에서 답변없이 에러 메시지만 표출해주세요'
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}" 
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()
    response_text = response_json['choices'][0]['message']['content']

    return response_text

# 엑셀 파일 불러오기
def load_excel(file_pth):
    df = pd.read_excel(file_pth)
    return df

# 데이터 전처리 및 임베딩
def preprocess_and_embed(df, question_column):
    # 텍스트 데이터를 문자열로 변환하고 결측값을 빈 문자열로 대체
    df[question_column] = df[question_column].astype(str).fillna('')
    
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    question_embeddings = model.encode(df[question_column].tolist(), convert_to_tensor=True)
    return model, question_embeddings

# FAISS 벡터화
def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)  # L2 distance
    index.add(np.array(embeddings))  # Add vectors to the index
    return index

# 에러메시지-답변 시스템
def answer_question(question, model, index, df, question_column, answer_column, top_k=1):
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), top_k)
    answers = df.iloc[I[0]][answer_column].tolist()
    return answers

# 파일 경로 및 텍스트 컬럼명 설정
file_pth = 'testdata.xlsx'  # 엑셀 파일 경로를 입력하세요
question_column = '에러메세지'    # 질문 텍스트 컬럼명을 입력하세요
answer_column = '답변'      # 답변 텍스트 컬럼명을 입력하세요

# 엑셀 파일 불러오기
df = load_excel(file_pth)

# 데이터 전처리 및 임베딩
model, question_embeddings = preprocess_and_embed(df, question_column)

# FAISS 벡터화
index = create_faiss_index(question_embeddings)

def get_answer(extracted_text):
    answer= answer_question(extracted_text,model, index, df, question_column,answer_column, top_k=1)
    return answer


#--------------------------------------- streamlit 구동

st.set_page_config(page_title="SAP 에러메세지 질의응답", page_icon=":robot:")
st.header("SAP 에러메세지 질의응답")

message = st.text_input('메시지를 입력하세요')
uploaded_file = st.file_uploader("image 파일을 선택하세요", type=['png','jpg'])

if uploaded_file:
    saved_file_path = save_uploaded_file('dataset', uploaded_file)
    if saved_file_path:
        st.success(f"'{uploaded_file.name}' 파일이 성공적으로 업로드 되었습니다.")
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        extracted_text = extract_text(saved_file_path)
        st.subheader("에러메시지")
        st.markdown(extracted_text, unsafe_allow_html=True)
        final_query = extracted_text
else:
    final_query = message

if final_query:
    answer = get_answer(final_query)
    st.subheader("해결방법")
    st.markdown(answer[0], unsafe_allow_html=True)
else:
    st.warning("메시지나 이미지 파일을 입력하세요.")