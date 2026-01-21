# app/main.py
from fastapi import FastAPI

app = FastAPI(title="Cars classifier")

@app.get("/")
def read_root():
    return {"message": "Service responds with success"}

# Test endpoint - to be removed
@app.get("/health")
def health_check():
    return {"status": "ok"}