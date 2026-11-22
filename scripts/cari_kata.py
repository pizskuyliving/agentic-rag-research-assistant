"""Cek apakah sebuah kata/istilah benar-benar ada di chunk yang tersimpan.

Berguna untuk memastikan masalahnya di RETRIEVAL (chunk ada tapi tidak ditemukan)
atau di INGESTION (teksnya memang tidak terbaca dari PDF).

Contoh:
    python -m scripts.cari_kata "Ambiguous" --user muhammad
    python -m scripts.cari_kata "ISREL" --user muhammad
"""
import argparse
import re

from app.ingestion.indexer import _semua_chunk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kata")
    ap.add_argument("--user", required=True)
    args = ap.parse_args()

    pola = re.compile(re.escape(args.kata), re.IGNORECASE)
    temuan = [c for c in _semua_chunk(args.user) if pola.search(c["teks"])]
    temuan.sort(key=lambda c: (c["penulis"], c["halaman"], c["urutan"]))

    print(f"'{args.kata}' ditemukan di {len(temuan)} chunk:\n")
    for c in temuan:
        m = pola.search(c["teks"])
        awal = max(0, m.start() - 60)
        cuplikan = " ".join(c["teks"][awal:m.end() + 60].split())
        print(f"- {c['penulis'][:25]:25} hal.{c['halaman']:>3} chunk#{c['urutan']:<4} ...{cuplikan}...")


if __name__ == "__main__":
    main()
