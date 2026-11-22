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
    tabel: list[str] = field(default_factory=list)   # tabel sebagai markdown, diawali "[TABEL] <judul>"


# Judul tabel di paper: "Table 1: ...", "Tabel 2. ..."
_POLA_JUDUL_TABEL = re.compile(r"^\s*(Table|Tabel)\s+[A-Z]?\d+\s*[:.]", re.IGNORECASE)


def _judul_tabel(blok: list, bbox) -> tuple[str, tuple | None]:
    """Cari judul tabel ("Table 1: ...") yang menempel/di dalam area tabel, atau paling dekat
    di atas/bawahnya (maks. 120 pt). Kembalikan (teks judul, bbox judul)."""
    x0, y0, x1, y1 = bbox
    terbaik, kotak, jarak_terbaik = "", None, 1e9
    for b in blok:
        bx0, by0, bx1, by1, teks = b[0], b[1], b[2], b[3], b[4]
        if not _POLA_JUDUL_TABEL.match(teks or ""):
            continue
        if bx1 < x0 or bx0 > x1:              # beda kolom halaman (paper dua kolom)
            continue
        if by1 >= y0 and by0 <= y1:           # judul tumpang-tindih dengan area tabel
            jarak = 0.0
        else:
            jarak = min(abs(by1 - y0), abs(by0 - y1))
        if jarak < jarak_terbaik and jarak < 120:
            terbaik, kotak, jarak_terbaik = " ".join(teks.split())[:250], (bx0, by0, bx1, by1), jarak
    return terbaik, kotak


def _ekstrak_tabel(page) -> list[str]:
    """Ambil isi tabel sebagai chunk tersendiri yang diawali judulnya.

    Dua pelajaran dari tes di paper Self-RAG:
    1. Memaksa tabel tanpa garis vertikal menjadi grid kolom memotong kata di tengah
       ("reflectio", "Fou") dan mencampur tabel dengan gambar. Jadi isi tabel diambil
       sebagai TEKS BIASA dari area tabel (kata tetap utuh), bukan sebagai grid.
    2. Gambar/contoh juga sering terdeteksi sebagai "tabel". Jadi area hanya dipakai jika
       ada judul "Table N:" / "Tabel N:" di dekatnya.
    Gagal mendeteksi tabel tidak boleh menggagalkan ingest.
    """
    try:
        temuan = page.find_tables(vertical_strategy="text", horizontal_strategy="lines")
        blok = page.get_text("blocks")
    except Exception:
        return []
    hasil, sudah = [], set()
    for t in getattr(temuan, "tables", []):
        try:
            judul, kotak_judul = _judul_tabel(blok, t.bbox)
            if not judul or judul in sudah:
                continue
            isi = _bersihkan(page.get_text("text", clip=t.bbox))
        except Exception:
            continue
        # judul sudah ditaruh di baris pertama; buang salinannya dari isi
        isi = isi.replace(judul, " ").strip() if judul in " ".join(isi.split()) else isi
        if len(re.sub(r"\s", "", isi)) < 40:
            continue
        sudah.add(judul)
        hasil.append(f"[TABEL] {judul}\n{isi}")
    return hasil


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
            tabel = _ekstrak_tabel(page)
            if len(teks) >= MIN_KARAKTER_PER_HALAMAN or tabel:
                halaman.append(Halaman(nomor=i, teks=teks, tabel=tabel))

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
    # Tes cepat:   python -m app.ingestion.pdf_parser data/papers/contoh.pdf
    # Lihat tabel: python -m app.ingestion.pdf_parser data/papers/contoh.pdf --tabel
    import sys
    j = baca_pdf(sys.argv[1])
    jumlah_tabel = sum(len(h.tabel) for h in j.halaman)
    print(f"ID      : {j.paper_id}\nJudul   : {j.judul}\nPenulis : {j.penulis}\nTahun   : {j.tahun}")
    print(f"Halaman : {j.jumlah_halaman} berisi teks | tabel terdeteksi: {jumlah_tabel} | "
          f"perlu dicek: {', '.join(j.perlu_dicek)}")
    print("-" * 60)
    if "--tabel" in sys.argv:
        for h in j.halaman:
            for t in h.tabel:
                print(f"\n=== halaman {h.nomor} ===\n{t[:1500]}")
    else:
        print(j.halaman[0].teks[:800])
