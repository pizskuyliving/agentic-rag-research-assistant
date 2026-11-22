"""Masukkan semua PDF di sebuah folder ke Qdrant.

Contoh:
    python -m scripts.ingest data/papers --user muhammad
"""
import argparse
import sys
from pathlib import Path

from app.ingestion.pdf_parser import PDFTidakTerbaca
from app.ingestion.pipeline import proses_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest PDF jurnal ke Qdrant")
    ap.add_argument("folder", help="folder berisi file PDF")
    ap.add_argument("--user", required=True, help="pemilik jurnal, misalnya: muhammad")
    args = ap.parse_args()

    files = sorted(Path(args.folder).glob("*.pdf"))
    if not files:
        sys.exit(f"Tidak ada PDF di {args.folder}")

    berhasil, gagal = 0, []
    for f in files:
        try:
            h = proses_pdf(f, user_id=args.user)
            cek = f"  (cek: {', '.join(h['perlu_dicek'])})" if h["perlu_dicek"] else ""
            print(f"     {h['judul'][:80]} | {h['penulis']} | {h['tahun']}")
            print(f"OK   {f.name}: {h['halaman']} hal, {h['chunk']} chunk{cek}")
            berhasil += 1
        except PDFTidakTerbaca as e:
            gagal.append(f.name)
            print(f"LEWAT {e}")
        except Exception as e:  # satu PDF rusak tidak boleh menghentikan semuanya
            gagal.append(f.name)
            print(f"GAGAL {f.name}: {e}")

    print(f"\nSelesai: {berhasil} jurnal masuk, {len(gagal)} dilewati.")


if __name__ == "__main__":
    main()
