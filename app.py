import streamlit as st
from transformers import pipeline

# Load model
@st.cache_resource
def load_model():
    ner = pipeline("ner", model="./ner_contract_extended", tokenizer="./ner_contract_extended", aggregation_strategy="simple")
    return ner

ner_model = load_model()

st.title("🔍 NER Trích Xuất Thực Thể trong Hợp Đồng")
st.write("Nhập vào đoạn văn bản để mô hình trích xuất các thực thể như Bên A, Bên B, Giá trị, Ngày hiệu lực, v.v.")

text = st.text_area("Nhập văn bản:", "Hợp đồng có hiệu lực từ ngày 01/01/2024 giữa Công ty A và Công ty B với giá trị 500 triệu đồng.")

if st.button("Phân tích"):
    if text.strip():
        results = ner_model(text)
        st.subheader("📌 Kết quả trích xuất:")
        for r in results:
            st.write(f"**{r['word']}** → `{r['entity_group']}` (score={r['score']:.2f})")
    else:
        st.warning("Vui lòng nhập nội dung hợp đồng!")
