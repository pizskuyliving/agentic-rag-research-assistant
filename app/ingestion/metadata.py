"""Ambil judul, penulis, dan tahun ARTIKEL dari halaman pertama memakai LLM ringan.

Metadata bawaan PDF sering kosong atau salah (misalnya judul berisi nama jurnal,
penulis "Un-named"). Membaca halaman pertama jauh lebih andal, dan hanya butuh
satu panggilan LLM per jurnal saat ingest.
"""
import json
import re

from app.llm import panggil_llm

PROMPT = """Berikut teks halaman pertama sebuah artikel ilmiah.
Ambil metadata ARTIKEL-nya, BUKAN nama jurnal, prosiding, atau penerbit.

Balas HANYA dengan JSON valid, tanpa penjelasan, dengan format:
{"judul": "judul lengkap artikel", "penulis": ["Nama Penulis 1", "Nama Penulis 2"], "tahun": 2024}

Jika ada yang tidak ditemukan, isi dengan null.

Teks halaman pertama:
"""


def _format_penulis(nama: list[str]) -> str:
    if len(nama) == 1:
        return nama[0]
    if len(nama) == 2:
        return f"{nama[0]} & {nama[1]}"
    return f"{nama[0]}, dkk."


def ekstrak_metadata(teks_halaman1: str) -> dict:
    """Kembalikan dict berisi kunci yang berhasil diambil: judul, penulis, tahun.
    Kalau LLM gagal atau jawabannya tidak valid, kembalikan dict kosong."""
    try:
        jawaban = panggil_llm(PROMPT + teks_halaman1[:3000], peran="ringan", max_tokens=400, json_mode=True)
    except Exception as e:
        print(f"     (metadata LLM gagal: {e})")
        return {}

    m = re.search(r"\{.*\}", jawaban, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}

    hasil = {}

    judul = data.get("judul")
    if isinstance(judul, str) and len(judul.strip()) >= 8:
        hasil["judul"] = " ".join(judul.split())   # rapikan spasi & baris baru

    penulis = data.get("penulis")
    if isinstance(penulis, str):
        penulis = [p for p in re.split(r",|\band\b|\bdan\b|&", penulis)]
    if isinstance(penulis, list):
        # buang angka afiliasi di belakang nama, mis. "Surya Hidayatullah Firdaus 1"
        nama = [re.sub(r"[\d*,]+$", "", p).strip() for p in penulis if isinstance(p, str)]
        nama = [n for n in nama if len(n) >= 3]
        if nama:
            hasil["penulis"] = _format_penulis(nama)

    tahun = data.get("tahun")
    try:
        tahun = int(tahun)
        if 1950 <= tahun <= 2100:
            hasil["tahun"] = tahun
    except (TypeError, ValueError):
        pass

    return hasil
