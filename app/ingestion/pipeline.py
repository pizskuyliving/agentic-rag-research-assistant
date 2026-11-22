"""Satu fungsi untuk memproses satu PDF dari awal sampai tersimpan di Qdrant.
Dipakai oleh scripts/ingest.py (terminal) dan app/main.py (API /upload)."""
from pathlib import Path

from app.ingestion.chunker import potong_jurnal
from app.ingestion.indexer import simpan_jurnal
from app.ingestion.metadata import ekstrak_metadata
from app.ingestion.pdf_parser import baca_pdf


def proses_pdf(path: str | Path, user_id: str) -> dict:
    """Baca -> perbaiki metadata -> potong -> embedding -> simpan.
    Bisa melempar PDFTidakTerbaca untuk PDF hasil scan."""
    jurnal = baca_pdf(path)

    # Perbaiki judul/penulis/tahun dengan membaca halaman pertama (LLM ringan)
    meta = ekstrak_metadata(jurnal.halaman[0].teks)
    for kunci, nilai in meta.items():
        setattr(jurnal, kunci, nilai)
    jurnal.perlu_dicek = [k for k in ("judul", "penulis", "tahun") if k not in meta]

    chunks = potong_jurnal(jurnal)
    n = simpan_jurnal(jurnal, chunks, user_id=user_id)

    return {
        "paper_id": jurnal.paper_id,
        "nama_file": jurnal.nama_file,
        "judul": jurnal.judul,
        "penulis": jurnal.penulis,
        "tahun": jurnal.tahun,
        "halaman": jurnal.jumlah_halaman,
        "chunk": n,
        "perlu_dicek": jurnal.perlu_dicek,
    }
