import pandas as pd
from fastapi.templating import Jinja2Templates
from models.emotion_model import SentimentAnalyzer

# 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="templates")

# 노래 데이터 로드
songs_data = pd.read_csv("data/songs_mood.csv")

# 감정 → Mood 매핑 (필터링용)
MOOD_COMMENTS = {
    "행복": "오늘 기분이 행복하군요! 신나는 노래로 더 즐겁게!",
    "놀람": "조금 놀란 하루였군요. 마음을 안정시켜줄 음악이에요.",
    "분노": "속상했던 하루였나요? 강렬한 비트로 날려버려요!",
    "공포": "불안한 감정에는 따뜻한 감성 음악이 어울려요.",
    "혐오": "기분이 좋지 않다면 차분한 음악으로 진정해보세요.",
    "슬픔": "마음이 무거우셨군요. 위로가 되는 곡들을 준비했어요.",
    "중립": "오늘은 담담한 하루, 잔잔한 곡들과 함께해요."
}

# 모델 초기화
analyzer = SentimentAnalyzer()

def analyze_text(text: str) -> str:
    return analyzer.analyze_sentiment(text)

def render_list_page(request, mood: str):
    try:
        filtered = songs_data[songs_data["Mood"] == mood]
        recommended = filtered.head(12).to_dict(orient="records")
        feedback = MOOD_COMMENTS.get(mood, "오늘의 추천입니다.")
        return templates.TemplateResponse("list.html", {
            "request": request,
            "songs": recommended,
            "mood": mood,
            "feedback": feedback
        })
    except Exception as e:
        return templates.TemplateResponse("list.html", {
            "request": request,
            "songs": [],
            "mood": mood,
            "message": "추천된 노래를 불러오는 데 실패했습니다.",
            "error": str(e)
        })

def render_box_page(request, mood: str, title: str):
    try:
        filtered = songs_data[songs_data["Mood"] == mood]
        selected = filtered.sample(n=min(8, len(filtered))).to_dict(orient="records")
        playlist = selected[:4]
        top_songs = selected[:5]
        return templates.TemplateResponse("box.html", {
            "request": request,
            "playlist": playlist,
            "top_songs": top_songs,
            "title": title,
            "mood": mood
        })
    except Exception as e:
        return templates.TemplateResponse("box.html", {
            "request": request,
            "error": str(e)
        })
