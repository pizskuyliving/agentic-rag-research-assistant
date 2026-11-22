"""Jalankan semua pertanyaan uji lewat agen, beri skor otomatis sederhana.

    python -m eval.run_eval --user muhammad

Skor otomatis hanya memeriksa hal yang bisa dicek mesin:
- apakah sumber wajib ikut disitasi, dan sumber terlarang tidak
- apakah kata kunci penting muncul di jawaban
- apakah pertanyaan di luar koleksi benar-benar ditolak tanpa sumber
Kebenaran isi tetap perlu dibaca manusia: jawabannya dicetak lengkap untuk itu.
"""
import argparse
import json
import re
import time
from pathlib import Path

from app.agent.graph import tanya_agen

FILE_SOAL = Path(__file__).parent / "questions.jsonl"


def _normal(teks: str) -> str:
    """Samakan variasi tanda hubung & spasi Unicode, mis. 'Sentence‑BERT' (U+2011) -> 'sentence-bert'."""
    teks = re.sub(r"[‐-―−]", "-", teks)
    teks = re.sub(r"[   ]", " ", teks)
    return teks.lower()


def _muncul(istilah: str, teks: str) -> bool:
    """Cocokkan sebagai kata utuh: 'correct' TIDAK cocok dengan 'corrective'."""
    return re.search(rf"(?<![\w-]){re.escape(_normal(istilah))}(?![\w])", teks) is not None


def nilai(soal: dict, hasil: dict) -> list[tuple[str, bool]]:
    jawaban = _normal(hasil["answer"])
    penulis = " ".join(s["penulis"] for s in hasil.get("citations", []))
    cek = []
    for nama in soal.get("sumber_wajib", []):
        cek.append((f"sumber wajib: {nama}", nama in penulis))
    for nama in soal.get("sumber_terlarang", []):
        cek.append((f"tanpa sumber: {nama}", nama not in penulis))
    for kata in soal.get("harus_menyebut", []):
        # "a|b" = cukup salah satu muncul (jawaban bisa Indonesia atau Inggris)
        pilihan = [k.strip() for k in kata.split("|")]
        cek.append((f"menyebut: {kata}", any(_muncul(k, jawaban) for k in pilihan)))
    if soal.get("route"):
        cek.append((f"router memilih {soal['route']}", hasil.get("route") == soal["route"]))
    if soal.get("harus_tidak_ada"):
        cek.append(("tanpa daftar sumber", not hasil.get("citations")))
        # menolak sungguhan: menyebut koleksi & tidak memberi angka dari ingatan model
        cek.append(("menyatakan tidak ada di koleksi", "koleksi" in jawaban))
        cek.append(("tidak mengarang angka", not re.search(r"\d", jawaban)))
    return cek


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--soal", default="",
                    help="jalankan soal tertentu saja, mis. --soal 5,6,7 (nomor mulai dari 1)")
    ap.add_argument("--jeda", type=int, default=30,
                    help="detik jeda antar soal, agar kuota token per menit Groq gratis tidak habis")
    args = ap.parse_args()

    pilihan = {int(x) for x in args.soal.split(",") if x.strip().isdigit()}
    semua_soal = [json.loads(b) for b in FILE_SOAL.read_text(encoding="utf-8").splitlines() if b.strip()]
    daftar = [(n, s) for n, s in enumerate(semua_soal, start=1) if not pilihan or n in pilihan]

    total_lulus = total_cek = 0
    rekap = []
    for i, (nomor, soal) in enumerate(daftar):
        if i > 0 and args.jeda:
            print(f"\n(jeda {args.jeda} detik untuk kuota per menit...)")
            time.sleep(args.jeda)
        hasil = tanya_agen(soal["pertanyaan"], user_id=args.user)
        cek = nilai(soal, hasil)
        lulus = sum(ok for _, ok in cek)
        total_lulus += lulus
        total_cek += len(cek)

        print("=" * 90)
        print(f"#{nomor} [{soal.get('jenis', '-')}] T: {soal['pertanyaan']}")
        print(f"Jejak: {' > '.join(hasil.get('trace', []))}")
        print(f"\n{hasil['answer']}\n")
        for s in hasil.get("citations", []):
            print(f"  [{s['nomor']}] {s['penulis']} ({s['tahun']}), hal. {s['halaman']}")
        print(f"\nSkor otomatis: {lulus}/{len(cek)}")
        for nama, ok in cek:
            print(f"  {'LULUS' if ok else 'GAGAL'}  {nama}")
        if soal.get("perlu_dicek"):
            print("Catatan: kunci jawaban soal ini disusun tanpa membaca jurnalnya; cek ke PDF.")
        rekap.append((nomor, soal.get("jenis", "-"), lulus, len(cek)))
        if soal.get("catatan_rag_dasar"):
            print(f"Pembanding (RAG dasar): {soal['catatan_rag_dasar']}")

    print("=" * 90)
    print("REKAP")
    for nomor, jenis, lulus, n in rekap:
        print(f"  #{nomor:<3} {lulus}/{n}  {jenis}")
    print(f"TOTAL skor otomatis: {total_lulus}/{total_cek}")


if __name__ == "__main__":
    main()
