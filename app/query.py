"""Multi-query + dekomposisi: tulis ulang pertanyaan sebelum mencari.

- Versi Bahasa Inggris & Indonesia: model embedding cenderung menganggap teks berbahasa
  sama lebih mirip, jadi dua versi memberi peluang ke jurnal kedua bahasa.
- Sub-query: pertanyaan perbandingan ("A vs B") dipecah per hal, supaya setiap
  hal mendapat chunk sendiri (pelajaran dari tes Self-RAG vs CRAG).
"""
import json
import re

from app.agent.prompts import QUERY
from app.llm import panggil_llm


def variasi_query(pertanyaan: str) -> list[str]:
    """Kembalikan daftar query unik: pertanyaan asli + EN + ID + sub-query (jika ada)."""
    hasil = [pertanyaan]
    try:
        jawaban = panggil_llm(QUERY + pertanyaan, peran="ringan", max_tokens=400, json_mode=True)
        m = re.search(r"\{.*\}", jawaban, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            kandidat = [data.get("en"), data.get("id")]
            sub = data.get("sub")
            if isinstance(sub, list):
                kandidat += sub[:3]
            # Istilah bagian metode (mirip HyDE): menjembatani kosakata pertanyaan dan kosakata paper.
            # Hanya dipakai untuk MENCARI; jawaban tetap berdasar chunk yang ditemukan.
            istilah = data.get("istilah_metode") or data.get("kata_kunci")
            if isinstance(istilah, str):
                istilah = [istilah]
            if isinstance(istilah, list):
                kandidat += istilah[:3]
            sudah = {h.lower() for h in hasil}
            for q in kandidat:
                if isinstance(q, str) and q.strip() and q.strip().lower() not in sudah:
                    hasil.append(q.strip())
                    sudah.add(q.strip().lower())
    except Exception as e:  # kalau gagal, tetap cari dengan pertanyaan asli
        print(f"(multi-query dilewati: {e})")
    return hasil
