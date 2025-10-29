import os
import gradio as gr
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import requests
import json
import re
from collections import deque
from dotenv import load_dotenv

# Load .env
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_PATH = os.getenv("MODEL_PATH", "emotion_model/models/transformer_model")

# Load model
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
emo_model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
emo_model.eval()

label_map_path = os.path.join(MODEL_PATH, "label_map.json")
if os.path.exists(label_map_path):
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    id_to_label = {int(v): k for k, v in label_map.items()}
else:
    id_to_label = {0: "joy", 1: "love", 2: "anger", 3: "fear", 4: "sadness", 5: "neutral"}

conversation_memory = deque(maxlen=5)

def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = emo_model(**inputs).logits
        pred = int(torch.argmax(logits, dim=-1).cpu().numpy()[0])
    return id_to_label.get(pred, "neutral")

def generate_reply(message):
    emotion = detect_emotion(message)
    payload = {
        "model": "meta-llama/llama-3.1-70b-instruct",
        "messages": [
            {"role": "system", "content": "You are a kind and empathetic mental health chatbot."},
            {"role": "user", "content": f"User feels {emotion}. Message: {message}"}
        ],
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=20)
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
        return f"({emotion}) {reply}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

iface = gr.Interface(
    fn=generate_reply,
    inputs=gr.Textbox(label="Type your message"),
    outputs=gr.Textbox(label="AI Reply"),
    title="🧠 Mental Health Companion",
    description="Chat safely with your AI companion for emotional support."
)

if __name__ == "__main__":
    iface.launch()
