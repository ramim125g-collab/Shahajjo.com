import os
import re
import json
import torch
import torch.nn as nn
import urllib.parse
import urllib.request
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from duckduckgo_search import DDGS

app = FastAPI(title="Shahajjo AI API", version="2.0")

# Enable CORS for Web UI / Mobile App Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "Vidclip/Shahajjo"
device = torch.device("cpu")

# ==========================================
# 1. LOAD MODEL & CLEAN SUBWORD VOCAB
# ==========================================
print("📥 Fetching Model & Vocab from Hugging Face...", flush=True)

try:
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json")
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        stoi = vocab_data.get("stoi", {})
        itos = {int(k): v for k, v in vocab_data.get("itos", {}).items()}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        d_model = config.get("d_model", 512)
        nhead = config.get("nhead", 8)
        num_layers = config.get("num_layers", 6)

    vocab_size = len(stoi)
    PAD_IDX = stoi.get("<pad>", 0)
    UNK_IDX = stoi.get("<unk>", 1)
    SOS_IDX = stoi.get("<sos>", 2)
    EOS_IDX = stoi.get("<eos>", 3)

except Exception as e:
    print(f"❌ Error loading Hugging Face files: {e}")

# Tokenizer
def tokenize_text(text):
    return re.findall(r'[\u0980-\u09FF]+|[a-zA-Z0-9]+|[^\s\w]', text)

def encode(text):
    tokens = tokenize_text(text)
    return [stoi.get(t, UNK_IDX) for t in tokens]

def decode(tokens):
    words = [itos.get(t, "<unk>") for t in tokens if t not in [PAD_IDX, SOS_IDX, EOS_IDX]]
    return " ".join(words)

# Transformer Architecture
class ShahajjoTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2048, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        src_key_padding_mask = (x == PAD_IDX)
        out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)
        return self.fc_out(out)

# Load Weights
model = ShahajjoTransformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==========================================
# 2. SEARCH ENGINES (DUCKDUCKGO & WIKIPEDIA)
# ==========================================
def search_duckduckgo(query: str):
    results_text = ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
            for r in results:
                results_text += f"{r.get('title', '')}: {r.get('body', '')} "
    except Exception as e:
        results_text = ""
    return results_text.strip()

def search_wikipedia(query: str):
    wiki_text = ""
    try:
        # Search Bangla Wikipedia
        encoded_q = urllib.parse.quote(query)
        url = f"https://bn.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={encoded_q}&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req).read().decode('utf-8')
        data = json.loads(res)
        pages = data.get("query", {}).get("pages", {})
        for k, v in pages.items():
            if k != "-1" and "extract" in v:
                wiki_text += v["extract"][:300] + " "
    except Exception:
        pass
    return wiki_text.strip()

# ==========================================
# 3. API ENDPOINTS
# ==========================================
class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"status": "online", "model": "Shahajjo AI", "version": "2.0"}

@app.post("/generate")
def generate_response(req: ChatRequest):
    user_prompt = req.prompt.strip()
    
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    search_triggered = False
    context_data = ""

    # Check if prompt starts with 'Search:' or 'search:'
    if user_prompt.lower().startswith("search:"):
        search_triggered = True
        actual_query = user_prompt[7:].strip()
        
        # Search DuckDuckGo + Wikipedia
        ddg_info = search_duckduckgo(actual_query)
        wiki_info = search_wikipedia(actual_query)
        
        context_data = f"{ddg_info} {wiki_info}".strip()
        if not context_data:
            context_data = "কোন তথ্য পাওয়া যায়নি।"

        # Combine with prompt
        full_text = f"User: Search: {actual_query} Info: {context_data[:150]} Shahajjo:"
    else:
        full_text = f"User: {user_prompt} Shahajjo:"

    # Encode & Predict using Shahajjo Model
    encoded_input = [SOS_IDX] + encode(full_text) + [EOS_IDX]
    input_tensor = torch.tensor([encoded_input], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_ids = torch.argmax(outputs, dim=-1)[0].tolist()

    generated_reply = decode(predicted_ids)

    # Fallback to direct context display if model output is too short on search
    if search_triggered and len(generated_reply) < 5:
        generated_reply = f"অনুসন্ধান ফলাফল: {context_data}"

    return {
        "prompt": user_prompt,
        "search_triggered": search_triggered,
        "response": generated_reply
    }
    
