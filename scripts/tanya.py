"""RAG DASAR (tanpa agen) dari terminal. Disimpan sebagai pembanding versi agen.

Contoh:
    python -m scripts.tanya "Metode apa yang dipakai untuk deteksi hoaks?" --user muhammad
Versi agen: python -m scripts.tanya_agen "..." --user muhammad
"""
import argparse

from app.agent.prompts import GENERATE_SYSTEM
from app.ingestion.indexer import cari
from app.llm import panggil_llm
from app.query import variasi_query


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pertanyaan")
    ap.add_argument("--user", required=True)
    args = ap.parse_args()

    queries = variasi_query(args.pertanyaan)
    print("Query pencarian:")
    for q in queries:
        print(f"  - {q}")
    print()
    sumber = cari(queries, user_id=args.user)
    if not sumber:
        print("Belum ada jurnal untuk user ini. Jalankan scripts.ingest dulu.")
        return

    konteks = "\n\n".join(
        f"[{i}] {s['penulis']} ({s['tahun']}), hal. {s['halaman']}: {s['teks']}"
        for i, s in enumerate(sumber, start=1)
    )
    jawaban = panggil_llm(
        f"Potongan jurnal:\n{konteks}\n\nPertanyaan: {args.pertanyaan}",
        peran="berat", system=GENERATE_SYSTEM,
    )

    print(jawaban)
    print("\nSumber:")
    for i, s in enumerate(sumber, start=1):
        print(f"[{i}] {s['penulis']} ({s['tahun']}), hal. {s['halaman']} - {s['judul'][:70]}  (skor {s['skor']})")


if __name__ == "__main__":
    main()
