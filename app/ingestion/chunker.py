"""Memotong teks jurnal menjadi chunk, tanpa kehilangan nomor halaman.

Chunk tidak pernah melewati batas halaman, supaya setiap chunk punya
satu nomor halaman yang pasti untuk sitasi.
"""
import re
from dataclasses import dataclass

from app import config
from app.ingestion.pdf_parser import Jurnal


@dataclass
class Chunk:
    paper_id: str
    urutan: int        # chunk ke-berapa dalam jurnal
    halaman: int
    teks: str


def _potong_teks(teks: str, ukuran: int, overlap: int) -> list[str]:
    """Potong per paragraf; paragraf digabung sampai mendekati `ukuran` karakter."""
    paragraf = [p.strip() for p in re.split(r"\n\s*\n", teks) if p.strip()]
    hasil, sekarang = [], ""
    for p in paragraf:
        # paragraf yang sangat panjang dipecah per kalimat
        potongan = [p] if len(p) <= ukuran else re.split(r"(?<=[.!?])\s+", p)
        for bagian in potongan:
            if sekarang and len(sekarang) + len(bagian) + 1 > ukuran:
                hasil.append(sekarang)
                # overlap: bawa ekor chunk sebelumnya agar konteks tidak terputus
                sekarang = sekarang[-overlap:] if overlap else ""
            sekarang = f"{sekarang} {bagian}".strip()
    if sekarang:
        hasil.append(sekarang)
    return hasil


def potong_jurnal(jurnal: Jurnal, ukuran: int = config.CHUNK_SIZE,
                  overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    chunks, urutan = [], 0
    for hal in jurnal.halaman:
        for teks in _potong_teks(hal.teks, ukuran, overlap):
            if len(teks) < 80:      # buang potongan terlalu pendek (nomor halaman, header)
                continue
            chunks.append(Chunk(paper_id=jurnal.paper_id, urutan=urutan, halaman=hal.nomor, teks=teks))
            urutan += 1
    return chunks
