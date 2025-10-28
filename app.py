import streamlit as st
from transformers import pipeline

# ==========================================================
# 1. HÀM LOAD MÔ HÌNH
# ==========================================================
# - Dùng Streamlit cache_resource để cache mô hình NER.
# - Giúp tránh việc load lại model mỗi khi người dùng tương tác (ấn nút, nhập text...),
#   vì model Hugging Face thường rất nặng, tốn vài trăm MB.
# ==========================================================
@st.cache_resource
def load_model():
    ner = pipeline(
        "ner",                           # Task: Named Entity Recognition
        model="./ner_model_contract",     # Đường dẫn model fine-tuned local (thư mục chứa config + weights)
        tokenizer="xlm-roberta-base",     # Tokenizer tương ứng
        aggregation_strategy="simple"     # Gộp các token liền nhau thuộc cùng thực thể
    )
    return ner

# Gọi hàm load_model() — model chỉ được load 1 lần duy nhất
ner_model = load_model()

# ==========================================================
# 2. GIAO DIỆN ỨNG DỤNG STREAMLIT
# ==========================================================
st.title("🔍 NER Trích Xuất Thực Thể (PER, ORG, LOC, VAL)")
st.write(
    "Nhập vào đoạn văn bản để mô hình trích xuất các thực thể như "
    "**Tên người (PER)**, **Tổ chức (ORG)**, **Địa điểm (LOC)** và **Giá trị (VAL)**."
)

# Ô nhập văn bản mẫu
text = st.text_area(
    "Nhập văn bản:",
    "Nguyễn Văn A làm việc tại Công ty ABC ở Hà Nội với mức lương 50 triệu đồng mỗi tháng."
)

# ==========================================================
# 3. BẢNG MÀU CHO CÁC LOẠI THỰC THỂ
# ==========================================================
# Giúp tô màu khác nhau cho từng nhãn (entity group)
ENTITY_COLORS = {
    "PER": "#ffadad",   # Màu đỏ nhạt cho tên người
    "ORG": "#ffd6a5",   # Màu cam nhạt cho tổ chức
    "LOC": "#9bf6ff",   # Màu xanh dương nhạt cho địa điểm
    "VAL": "#caffbf"    # Màu xanh lá nhạt cho giá trị/số tiền
}

# ==========================================================
# 4. HÀM HIGHLIGHT THỰC THỂ TRONG VĂN BẢN
# ==========================================================
# - Nhận vào văn bản gốc và danh sách thực thể (entities)
# - Duyệt qua từng thực thể có độ tin cậy > 0.8
# - Chèn thẻ <span> HTML với background-color tương ứng
# ==========================================================
def highlight_entities(text, entities):
    highlighted = ""
    last_idx = 0  # Giữ vị trí của đoạn văn đã xử lý xong

    for ent in entities:
        if ent["score"] < 0.8:
            continue  # Bỏ qua thực thể không đủ tin cậy

        start, end = ent["start"], ent["end"]
        label = ent["entity_group"]
        color = ENTITY_COLORS.get(label, "#e0e0e0")  # Mặc định màu xám nếu không có trong bảng màu

        # Thêm phần văn bản trước thực thể
        highlighted += text[last_idx:start]

        # Thêm phần thực thể được tô màu
        highlighted += f"<span style='background-color:{color}; padding:2px 4px; border-radius:4px;'>{text[start:end]}</span>"

        last_idx = end  # Cập nhật vị trí

    # Thêm phần văn bản còn lại
    highlighted += text[last_idx:]
    return highlighted

# ==========================================================
# 5. HÀM HIỂN THỊ CHÚ GIẢI MÀU
# ==========================================================
# - Giúp người dùng biết mỗi màu tương ứng với loại thực thể nào
# ==========================================================
def render_legend():
    legend_html = "<div style='display:flex; flex-wrap:wrap; gap:10px; margin-top:10px;'>"
    for label, color in ENTITY_COLORS.items():
        legend_html += (
            f"<div style='display:flex; align-items:center; gap:6px;'>"
            f"<div style='width:16px; height:16px; background-color:{color}; border-radius:3px;'></div>"
            f"<span style='font-size:14px;'>{label}</span>"
            f"</div>"
        )
    legend_html += "</div>"
    return legend_html

# ==========================================================
# 6. NÚT “PHÂN TÍCH”
# ==========================================================
# Khi người dùng bấm nút:
# - Gọi pipeline NER để trích xuất thực thể
# - Lọc theo ngưỡng score > 0.8
# - Hiển thị kết quả tô màu và bảng chú giải
# ==========================================================
if st.button("Phân tích"):
    if text.strip():  # Kiểm tra có nhập nội dung không
        # Gọi mô hình để nhận diện thực thể
        results = ner_model(text)

        # Lọc bỏ các thực thể có độ tin cậy thấp
        filtered = [r for r in results if r["score"] >= 0.8]

        if filtered:
            # Hiển thị văn bản có highlight
            st.markdown("### Văn bản có highlight:")
            st.markdown(highlight_entities(text, filtered), unsafe_allow_html=True)

            # Hiển thị bảng chú giải
            st.markdown("### Chú giải các loại thực thể:")
            st.markdown(render_legend(), unsafe_allow_html=True)
        else:
            st.info("❕Không có thực thể nào đạt độ tin cậy > 0.8.")
    else:
        st.warning("⚠️ Vui lòng nhập nội dung để phân tích!")

