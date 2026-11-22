"""Tanya lewat AGEN LangGraph (router -> retrieve -> grade -> generate).

Contoh:
    python -m scripts.tanya_agen "Apa perbedaan Self-RAG dan CRAG?" --user muhammad
    python -m scripts.tanya_agen "Apa perbedaan Self-RAG dan CRAG?" --user muhammad --detail

--detail menampilkan SEMUA kandidat dari retrieval dan keputusan Grade untuk masing-masing,
supaya kita bisa melihat sendiri: apakah chunk yang dibutuhkan ikut terambil? Apakah Grade
membuangnya? Ini cara mendiagnosis RAG tanpa menebak.
"""
import argparse

from app.agent.graph import tanya_agen


def _cetak_detail(hasil: dict) -> None:
    kandidat = hasil.get("chunks", [])
    relevan = {(c["paper_id"], c["urutan"]) for c in hasil.get("relevant", [])}
    print(f"\nKandidat retrieval ({len(kandidat)}), urut skor gabungan; LOLOS = diloloskan Grade:")
    for i, c in enumerate(kandidat, start=1):
        tanda = "LOLOS" if (c["paper_id"], c["urutan"]) in relevan else "  -  "
        cuplikan = " ".join(c["teks"].split())[:110]
        print(f" {i:2}. {tanda} {c['penulis'][:22]:22} hal.{c['halaman']:>3} "
              f"rrf={c.get('rrf', 0):.4f} vektor={c.get('skor', 0):.3f}")
        print(f"            \"{cuplikan}...\"")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pertanyaan")
    ap.add_argument("--user", required=True)
    ap.add_argument("--detail", action="store_true", help="tampilkan semua kandidat & keputusan Grade")
    args = ap.parse_args()

    hasil = tanya_agen(args.pertanyaan, user_id=args.user)

    print("Jejak agen:")
    for langkah in hasil.get("trace", []):
        print(f"  > {langkah}")
    if hasil.get("queries"):
        print("Query pencarian:")
        for q in hasil["queries"]:
            print(f"  - {q}")
    if args.detail:
        _cetak_detail(hasil)
    print("\n" + hasil["answer"])

    if hasil.get("citations"):
        print("\nSumber:")
        for s in hasil["citations"]:
            print(f"[{s['nomor']}] {s['penulis']} ({s['tahun']}), hal. {s['halaman']} - {s['judul'][:70]}")
    elif not hasil.get("from_collection"):
        print("\n(Jawaban ini tidak berasal dari koleksi jurnalmu.)")


if __name__ == "__main__":
    main()
