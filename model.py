# ======================== #
# 1. Import thư viện cần thiết
# ======================== #
# Thư viện xử lý và tải dữ liệu
from datasets import Dataset, load_dataset

# Đánh giá mô hình bằng các metrics như Accuracy, F1, Precision, Recall
import evaluate

from transformers import (
    AutoTokenizer,                      # Tokenizer tương thích với mô hình
    AutoModelForTokenClassification,    # Mô hình phân loại token
    TrainingArguments,                  # Các tham số cho quá trình huấn luyện
    Trainer,                            # Lớp huấn luyện
    DataCollatorForTokenClassification   # Bộ sưu tập dữ liệu cho phân loại token
)

# Thư viện tính toán số học (mảng, ma trận)
import numpy as np

# ======================== #
# 2. Tải và chia dữ liệu train/test
# ======================== #
datasets = load_dataset("json", data_files="/content/ner_bio_3.json")

split_dataset = datasets["train"].train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]


# ======================== #
# 3. Xử lý nhãn BIO
# ======================== #
labels = sorted(list({label for row in train_dataset for label in row["ner_tags"]}))
label2id = {label: i for i, label in enumerate(labels)} # Tạo từ điển ánh xạ nhãn sang chỉ số
id2label = {i: label for label, i in label2id.items()} # Tạo từ điển ánh xạ chỉ số sang nhãn

# ======================== #
# 4. Khởi tạo Tokenizer từ mô hình XLM-Roberta
# ======================== #
model_name = "xlm-roberta-base"  # Mô hình pre-trained gốc
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


# ======================== #
# 5. Hàm token hóa và căn chỉnh nhãn
# ======================== #
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    all_labels = []
    for i, labels_per_example in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Chỉ số -100 sẽ được bỏ qua trong tính toán loss
            else:
                label_ids.append(label2id[labels_per_example[word_idx]])
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

# ======================== #
# 6. Tokenize toàn bộ dataset
# ======================== #
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True, remove_columns=datasets["train"].column_names)
tokenized_datasets_test = test_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=test_dataset.column_names)


# ======================== #
# 7. Load mô hình
# ======================== #
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

# ======================== #
# 8. Setup Trainer
# ======================== #
data_collator = DataCollatorForTokenClassification(tokenizer) # Bộ sưu tập dữ liệu cho phân loại token
metric = evaluate.load("seqeval") # Sử dụng bộ đánh giá seqeval cho NER

# Hàm tính các metrics đánh giá như precision, recall, f1, accuracy
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# ======================== #
# 9. Cấu hình training
# ======================== #
training_args = TrainingArguments(
    output_dir="./ner_contract_model",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    report_to="none",
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch",
    fp16=True,  # Bật True nếu bạn dùng GPU hỗ trợ
)


# ======================== #
# 10. Huấn luyện & đánh giá mô hình
# ======================== #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate(tokenized_datasets_test)
print(results)