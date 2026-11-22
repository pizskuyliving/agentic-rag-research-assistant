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
from app.llm import KuotaHarianHabis

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
    ap.add_argument("--ulang", type=int, default=1,
                    help="jalankan seluruh soal N kali lalu rata-ratakan (LLM tidak selalu konsisten)")
    ap.add_argument("--jeda", type=int, default=30,
                    help="detik jeda antar soal, agar kuota token per menit Groq gratis tidak habis")
    args = ap.parse_args()

    pilihan = {int(x) for x in args.soal.split(",") if x.strip().isdigit()}
    semua_soal = [json.loads(b) for b in FILE_SOAL.read_text(encoding="utf-8").splitlines() if b.strip()]
    daftar = [(n, s) for n, s in enumerate(semua_soal, start=1) if not pilihan or n in pilihan]

    semua_rekap = []
    for putaran in range(1, args.ulang + 1):
        if args.ulang > 1:
            print(f"\n{'#' * 30} PUTARAN {putaran}/{args.ulang} {'#' * 30}")
        rekap = _jalankan(daftar, args)
        semua_rekap.append(rekap)
        if len(rekap) < len(daftar):      # terhenti (kuota habis): jangan mulai putaran berikutnya
            break

    if args.ulang > 1 and len(semua_rekap) > 1:
        print("=" * 90)
        print(f"RATA-RATA {args.ulang} PUTARAN")
        for idx, (nomor, jenis, _, n) in enumerate(semua_rekap[0]):
            nilai_soal = [r[idx][2] for r in semua_rekap if idx < len(r)]
            print(f"  #{nomor:<3} {'/'.join(map(str, nilai_soal))} dari {n}  {jenis}")
        lengkap = [r for r in semua_rekap if len(r) == len(semua_rekap[0])]
        if len(lengkap) < len(semua_rekap):
            print("  (putaran yang terhenti di tengah tidak ikut dirata-rata)")
        totals = [sum(x[2] for x in r) for r in lengkap]
        maks = sum(x[3] for x in semua_rekap[0])
        if not totals:
            return
        rata = sum(totals) / len(totals)
        print(f"TOTAL per putaran: {totals} dari {maks} | rata-rata {rata:.1f}/{maks} = {100 * rata / maks:.1f}%")


def _jalankan(daftar: list, args) -> list:
    total_lulus = total_cek = 0
    rekap = []
    for i, (nomor, soal) in enumerate(daftar):
        if i > 0 and args.jeda:
            print(f"\n(jeda {args.jeda} detik untuk kuota per menit...)")
            time.sleep(args.jeda)
        try:
            hasil = tanya_agen(soal["pertanyaan"], user_id=args.user)
        except KuotaHarianHabis as e:
            print("=" * 90)
            print(f"BERHENTI di soal #{nomor}: {e}")
            print("Rekap di bawah hanya untuk soal yang sudah selesai.")
            break
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
    print(f"TOTAL skor otomatis: {total_lulus}/{total_cek} = {100 * total_lulus / max(total_cek, 1):.1f}%")
    return rekap


if __name__ == "__main__":
    main()
