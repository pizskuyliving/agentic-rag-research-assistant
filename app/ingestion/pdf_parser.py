"""Membaca PDF jurnal: teks per halaman + metadata dasar.

Nomor halaman disimpan karena dipakai untuk sitasi "(Penulis, Tahun, hal. X)".
"""
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (nama impor baru)

# Halaman dengan teks lebih sedikit dari ini dianggap kosong / gambar.
MIN_KARAKTER_PER_HALAMAN = 50

# Isi metadata "author" yang sebenarnya berarti kosong
PENULIS_PLACEHOLDER = {"", "un-named", "unnamed", "unknown", "anonymous", "admin", "user", "author"}

# Baris yang biasanya header jurnal, bukan judul artikel
_POLA_HEADER = re.compile(r"\b(jurnal|journal|prosiding|proceedings?|vol\.?|volume|issn|doi|arxiv|seminar)\b",
                          re.IGNORECASE)


def _mirip_header(teks: str) -> bool:
    return bool(_POLA_HEADER.search(teks))


@dataclass
class Halaman:
    nomor: int          # mulai dari 1, sesuai yang dilihat pembaca
    teks: str


@dataclass
class Jurnal:
    paper_id: str
    nama_file: str
    judul: str
    penulis: str
    tahun: int | None
    halaman: list[Halaman] = field(default_factory=list)
    perlu_dicek: list[str] = field(default_factory=list)   # metadata yang tebakan

    @property
    def jumlah_halaman(self) -> int:
        return len(self.halaman)


class PDFTidakTerbaca(Exception):
    """PDF hasil scan / gambar: teksnya tidak bisa diambil tanpa OCR."""


def _bersihkan(teks: str) -> str:
    teks = teks.replace("­", "")                 # soft hyphen
    teks = re.sub(r"-\n(\w)", r"\1", teks)            # kata terpotong di akhir baris: "pene-\nlitian"
    teks = re.sub(r"[ \t]+", " ", teks)
    teks = re.sub(r"\n{3,}", "\n\n", teks)
    return teks.strip()


def _tebak_tahun(metadata: dict, teks_awal: str) -> int | None:
    # 1) cari tahun 19xx/20xx di halaman pertama (biasanya ada di header jurnal)
    tahun = re.findall(r"\b(19[89]\d|20[0-4]\d)\b", teks_awal[:3000])
    if tahun:
        return int(max(set(tahun), key=tahun.count))
    # 2) fallback: tanggal pembuatan file PDF, format "D:20230415..."
    m = re.match(r"D:(\d{4})", metadata.get("creationDate") or "")
    return int(m.group(1)) if m else None


def baca_pdf(path: str | Path) -> Jurnal:
    path = Path(path)
    data = path.read_bytes()
    paper_id = hashlib.sha1(data).hexdigest()[:12]   # file yang sama -> id yang sama

    with fitz.open(stream=data, filetype="pdf") as doc:
        meta = doc.metadata or {}
        halaman = []
        for i, page in enumerate(doc, start=1):
            teks = _bersihkan(page.get_text("text"))
            if len(teks) >= MIN_KARAKTER_PER_HALAMAN:
                halaman.append(Halaman(nomor=i, teks=teks))

    if not halaman:
        raise PDFTidakTerbaca(f"{path.name}: tidak ada teks terbaca (kemungkinan PDF hasil scan)")

    # Ini tebakan cepat tanpa LLM. scripts/ingest.py memperbaikinya lewat app/ingestion/metadata.py
    perlu_dicek = []
    judul = (meta.get("title") or "").strip()
    if len(judul) < 8 or _mirip_header(judul):
        # pakai baris panjang pertama di halaman 1 yang bukan header jurnal
        baris = [b.strip() for b in halaman[0].teks.splitlines()
                 if len(b.strip()) > 20 and not _mirip_header(b)]
        judul = baris[0] if baris else path.stem
        perlu_dicek.append("judul")

    penulis = (meta.get("author") or "").strip()
    if penulis.lower() in PENULIS_PLACEHOLDER:
        penulis = "Tanpa nama penulis"
        perlu_dicek.append("penulis")

    tahun = _tebak_tahun(meta, halaman[0].teks)
    perlu_dicek.append("tahun")  # selalu tebakan, sebaiknya dikonfirmasi pengguna

    return Jurnal(paper_id=paper_id, nama_file=path.name, judul=judul, penulis=penulis,
                  tahun=tahun, halaman=halaman, perlu_dicek=perlu_dicek)


if __name__ == "__main__":
    # Tes cepat: python -m app.ingestion.pdf_parser data/papers/contoh.pdf
    import sys
    j = baca_pdf(sys.argv[1])
    print(f"ID      : {j.paper_id}\nJudul   : {j.judul}\nPenulis : {j.penulis}\nTahun   : {j.tahun}")
    print(f"Halaman : {j.jumlah_halaman} berisi teks | perlu dicek: {', '.join(j.perlu_dicek)}")
    print("-" * 60)
    print(j.halaman[0].teks[:800])
