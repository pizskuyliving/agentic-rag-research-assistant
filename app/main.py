"""API FastAPI.

Jalankan:   uvicorn app.main:app --reload
Coba di:    http://localhost:8000/docs

Catatan: user_id masih dikirim manual. Saat dipakai beberapa orang,
ganti dengan login (misalnya Google) supaya user_id tidak bisa dipalsukan.
"""
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.agent.graph import tanya_agen
from app.ingestion.pdf_parser import PDFTidakTerbaca
from app.ingestion.pipeline import proses_pdf

app = FastAPI(title="Asisten Riset & Skripsi")

MAKS_UKURAN_MB = 20


class ChatRequest(BaseModel):
    question: str
    user_id: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload(user_id: str = Form(...), file: UploadFile = File(...)) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Hanya file PDF yang diterima")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(file.filename).name
        with path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        if path.stat().st_size > MAKS_UKURAN_MB * 1024 * 1024:
            raise HTTPException(413, f"Ukuran maksimal {MAKS_UKURAN_MB} MB")
        try:
            # proses berat (embedding) dijalankan di thread agar server tetap responsif
            return await run_in_threadpool(proses_pdf, path, user_id)
        except PDFTidakTerbaca as e:
            raise HTTPException(422, str(e))


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(400, "Pertanyaan kosong")
    hasil = await run_in_threadpool(tanya_agen, req.question, req.user_id)
    return {
        "answer": hasil["answer"],
        "from_collection": hasil.get("from_collection", False),
        "citations": [
            {k: s.get(k) for k in ("nomor", "penulis", "tahun", "halaman", "judul", "teks", "nama_file")}
            for s in hasil.get("citations", [])
        ],
        "trace": hasil.get("trace", []),
    }
