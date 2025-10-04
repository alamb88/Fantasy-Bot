from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from index import VideoIndex, answer

app = FastAPI()
vi = VideoIndex()
vi.load()

@app.get("/ask", response_class=PlainTextResponse)
def ask(q: str = Query(...)):
    hits = vi.search(q, k=5)
    return answer(q, hits)
