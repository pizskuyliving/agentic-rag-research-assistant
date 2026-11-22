"""Diagnosa jangkar entitas: chunk mana dari paper asli tiap metode yang dijamin ikut ke Generate.

Contoh:
    python -m scripts.cek_entitas --user muhammad "Apa perbedaan pendekatan Self-RAG dan CRAG dalam menangani dokumen yang tidak relevan?"

Secara default memakai variasi query dari LLM ringan (1 panggilan kecil) supaya sama persis
dengan agen. Tambahkan --tanpa-llm untuk memakai pertanyaan asli saja (tanpa kuota).
"""
import argparse

from app.agent.nodes import _kelas_jangkar, _nama_metode, ambil_jangkar
from app.ingestion.indexer import struktur_paper

KELAS = {0: "bagian metode", 1: "pendahuluan/related work/hasil/kesimpulan", 2: "referensi/lampiran/tabel angka"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pertanyaan")
    ap.add_argument("--user", required=True)
    ap.add_argument("--tanpa-llm", action="store_true")
    a = ap.parse_args()

    if a.tanpa_llm:
        queries = [a.pertanyaan]
    else:
        from app.query import variasi_query
        queries = variasi_query(a.pertanyaan)
    print(f"Nama metode terdeteksi: {_nama_metode(a.pertanyaan)}")
    print("Query:")
    for q in queries:
        print(f"  - {q}")
    print()
    for c in ambil_jangkar(a.pertanyaan, queries, a.user):
        kelas = KELAS[_kelas_jangkar(c, struktur_paper(a.user, c["paper_id"]))]
        print(f"{c['penulis']} hal. {c['halaman']} (urutan {c['urutan']}, {kelas}):")
        print(f"  {' '.join(c['teks'].split())[:220]}\n")


if __name__ == "__main__":
    main()
