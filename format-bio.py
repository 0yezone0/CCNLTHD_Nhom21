import json
import re

def convert_labelstudio_to_bio_json(input_path, output_path):
    # Đọc dữ liệu Label Studio
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = []

    for item in data:
        text = item["text"]
        entities = []
        for ent in item.get("label", []):
            entities.append({
                "start": ent["start"],
                "end": ent["end"],
                "label": ent["labels"][0]
            })

        # Tách token bằng regex để giữ vị trí chính xác
        tokens = []
        for match in re.finditer(r"\S+", text):
            token = match.group()
            start = match.start()
            end = match.end()
            tokens.append((token, start, end))

        labels = ["O"] * len(tokens)

        # Gán nhãn B- I-
        for ent in entities:
            found = False
            for i, (tok, s, e) in enumerate(tokens):
                if s >= ent["start"] and e <= ent["end"]:
                    if not found:
                        labels[i] = f"B-{ent['label']}"
                        found = True
                    else:
                        labels[i] = f"I-{ent['label']}"

        # Thêm vào danh sách kết quả
        result.append({
            "tokens": [tok for tok, _, _ in tokens],
            "ner_tags": labels
        })

    # Ghi ra file JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Đã xuất file BIO JSON tại: {output_path}")

convert_labelstudio_to_bio_json(
    input_path="project-13-at-2025-10-23-22-17-ecc497a3.json",   # file Label Studio của bạn
    output_path="ner_bio_3.json"       # file kết quả
)