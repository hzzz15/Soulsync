import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalyzer:
    def __init__(self):
        # Hugging Face에 업로드된 파인튜닝된 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained("hzz15/soulsync")
        self.model = AutoModelForSequenceClassification.from_pretrained("hzz15/soulsync")
        self.model.eval()
        
        self.emotion_labels = ["행복", "놀람", "분노", "공포", "혐오", "슬픔", "중립"]

    def analyze_sentiment(self, text: str) -> str:
        if not text.strip():
            raise ValueError("입력된 텍스트가 비어 있습니다.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(dim=1).item()

        return self.emotion_labels[predicted_label]
