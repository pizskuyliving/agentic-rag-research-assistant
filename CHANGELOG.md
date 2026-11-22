# Catatan perubahan

## Paket 7 — Tabel + evaluasi pertanyaan daftar (22 Nov 2026)

Evaluasi 15 soal sebelum paket ini: 48/56 (86%). Kegagalan terbesar #10 (jenis reflection token Self-RAG, 1/5):
rinciannya ada di Tabel 1 hal. 4 yang tidak pernah terambil, dan Evaluasi puas dengan jawaban kategori umum.

| File | Perubahan |
|---|---|
| `app/ingestion/pdf_parser.py` | Deteksi tabel per halaman (`find_tables`), simpan sebagai markdown + judul ("Table 1: ..."); `--tabel` untuk pratinjau |
| `app/ingestion/chunker.py` | Tabel = chunk `[TABEL]` tersendiri setelah teks halamannya; tabel besar dipecah per baris dengan header diulang |
| `app/ingestion/indexer.py` | Ingest ulang menghapus chunk lama jurnal itu dulu (nomor urut bisa bergeser) |
| `app/agent/prompts.py` | Grade: tabel berisi jawaban = relevan. Evaluasi: pertanyaan daftar cukup hanya jika item disebut spesifik |
| `eval/run_eval.py` | `--ulang N`: rata-rata beberapa putaran; persentase ditampilkan |

**Wajib ingest ulang** setelah paket ini.

Revisi 7b: tabel diambil sebagai teks biasa dari area tabel + wajib ada judul "Table N" (versi grid memotong kata).
Revisi 7c: pertanyaan daftar ("apa saja", "jenis", "types of") otomatis juga mencari di chunk tabel (BM25 khusus
tabel, `cari_tabel()`); BM25 menyamakan bentuk jamak (tokens = token). Evaluasi LLM mengabaikan aturan daftar di
prompt, jadi diganti logika pasti. docker-compose memakai volume Docker (bind mount ke D:\ error I/O di Windows).

## Paket 6 — Cek sitasi deterministik (20 Nov 2026)

Evaluasi setelah context expansion: total 13/19 (tertinggi), TAPI dua jawaban tanpa sitasi sama sekali
(konteks 9–13 chunk membuat model mengabaikan aturan sitasi).

| File | Perubahan |
|---|---|
| `app/agent/nodes.py` | Setelah Generate, kode menghitung kalimat klaim (>= 60 karakter) yang bersitasi. Jika < 70%, jawaban dikembalikan ke model SATU kali beserta daftar kalimat tanpa sitasi; versi baru dipakai hanya jika lebih baik. `[1, 3]` dan `【1】` dirapikan. |
| `app/agent/prompts.py` | `PENGINGAT_SITASI` (ditaruh di akhir pesan) dan `PERBAIKI_SITASI` (klaim tanpa dukungan dihapus) |

## Paket 5 — Grade dengan model berat + context expansion (20 Nov 2026)

| File | Perubahan | Kenapa |
|---|---|---|
| `app/config.py`, `.env.example` | `PERAN_GRADE` / `PERAN_EVALUASI` (default `berat`) | Model ringan sering salah menilai relevansi |
| `app/ingestion/indexer.py` | `ambil_chunk()`: ambil chunk tertentu per nomor urut | Untuk context expansion |
| `app/agent/nodes.py` | Generate menambahkan chunk sebelum & sesudah setiap chunk relevan (maks. ~12.000 karakter); kandidat 12, Grade membaca 750 karakter | Penjelasan sebuah ide sering berlanjut ke paragraf berikutnya; prompt "pakai istilah dari dokumen" diabaikan model, jadi diganti logika pasti |

## Paket 4 — Node Evaluasi + Rewrite (20 Nov 2026)

Temuan dari `--detail` dan `cari_kata`: teks metode CRAG (hal. 2–5) dan Self-RAG (hal. 4–5) ADA di database,
tapi tidak pernah terambil karena kosakata pertanyaan ("irrelevant documents") berbeda dengan kosakata paper
("retrieval evaluator", "reflection tokens"). Tebakan istilah dari ingatan model meleset.

| File | Perubahan |
|---|---|
| `app/agent/nodes.py` | Node baru `evaluasi`: cek apakah setiap aspek pertanyaan sudah terjelaskan; jika belum, tulis query baru memakai istilah yang MUNCUL di chunk yang sudah ditemukan, lalu cari ulang (maks. 1 putaran). Grade mengakumulasi hasil antar-putaran; Grade membaca 900 karakter per chunk |
| `app/agent/graph.py` | `grade -> evaluasi -> (generate / retrieve lagi / tidak_ada)` |
| `app/agent/state.py` | `putaran`, `query_baru`, `keputusan` |
| `app/agent/prompts.py` | Prompt `EVALUASI` baru; Grade: penjelasan mekanisme metode = relevan, daftar isi & detail eksperimen = tidak |
| `scripts/tanya_agen.py` | `--detail`: tampilkan semua kandidat + keputusan Grade |
| `scripts/cari_kata.py` | Cek apakah sebuah istilah ada di chunk tersimpan |

## Paket 3 — Perbaikan setelah evaluasi pertama (20 Nov 2026)

| File | Perubahan | Kenapa |
|---|---|---|
| `app/ingestion/bm25.py` (baru) | Pencarian kata kunci BM25 per user, indeks di memori, dibangun ulang otomatis saat ada ingest baru | Istilah persis (CRAG, ISREL, Self-RAG) sering terlewat oleh pencarian vektor |
| `app/ingestion/indexer.py` | `cari()` = hybrid: vektor + BM25 untuk setiap query, digabung dengan Reciprocal Rank Fusion | Chunk CRAG tentang aksi Correct/Incorrect/Ambiguous tidak pernah terambil |
| `app/agent/prompts.py` | Query: sub-query WAJIB untuk perbandingan + `kata_kunci`; Router: direct hanya sapaan/cara pakai; Direct dilarang memberi fakta | Sub-query kadang tidak dibuat; soal H100 sempat dijawab "$15.000" tanpa sumber |
| `app/query.py` | Menambahkan `kata_kunci` sebagai query tambahan | Untuk BM25 |
| `app/llm.py`, `app/config.py`, `.env.example` | Model ringan → `openai/gpt-oss-20b` (llama-3.1-8b-instant dimatikan Groq 16 Agt 2026); `reasoning_effort=low`; `max_tokens` minimal 600; tunggu & ulang otomatis saat 429; JSON mode | Model lama 404; jawaban kosong karena token habis untuk "berpikir"; batas TPM |
| `app/agent/nodes.py` | Parser JSON Grade yang toleran + balasan mentah dicatat di jejak saat gagal | Grade sempat selalu gagal |
| `eval/run_eval.py`, `eval/questions.jsonl` | Pencocokan kata utuh & tanda hubung Unicode; cek "tidak mengarang angka"; cek pilihan router; soal ke-4 (sapaan); jeda antar soal | Penilai sempat salah lulus/gagal |

Skor evaluasi sebelum hybrid search: 12/19 (PPDB 5/5, H100 4/4, sapaan 2/2, Self-RAG vs CRAG 1/8).

## Paket 2 — Minggu 3: agen LangGraph + API (20 Nov 2026)

### Baru
| File | Isi |
|---|---|
| `app/agent/state.py` | `ResearchState`: data yang dibawa antar-node |
| `app/agent/prompts.py` | Semua prompt (router, query, grade, generate, direct) di satu tempat |
| `app/agent/nodes.py` | Node: `router`, `retrieve`, `grade`, `generate`, `jawab_langsung`, `tidak_ada` |
| `app/agent/graph.py` | Graf LangGraph + fungsi `tanya_agen()` |
| `app/ingestion/metadata.py` | Ambil judul/penulis/tahun artikel dari halaman 1 pakai LLM ringan |
| `app/ingestion/pipeline.py` | `proses_pdf()`: satu fungsi ingest, dipakai terminal dan API |
| `app/query.py` | Multi-query EN/ID + pemecahan pertanyaan perbandingan (sub-query) |
| `app/main.py` | API FastAPI: `POST /upload`, `POST /chat`, `GET /health` |
| `scripts/tanya_agen.py` | Tanya lewat agen dari terminal, menampilkan jejak setiap node |
| `eval/questions.jsonl` | 3 pertanyaan uji + catatan hasil RAG dasar |
| `eval/run_eval.py` | Jalankan semua soal uji, skor otomatis sederhana |

### Diubah
| File | Perubahan | Kenapa |
|---|---|---|
| `app/ingestion/pdf_parser.py` | `import pymupdf as fitz`; lewati baris header jurnal saat menebak judul; "Un-named" dianggap kosong | Peringatan deprecated; judul sempat terbaca sebagai nama jurnal |
| `app/ingestion/indexer.py` | `cari()` menerima beberapa query, gabung hasil, maks. N chunk per jurnal | Paper berbahasa Inggris kalah dari jurnal Indonesia; satu jurnal mendominasi |
| `scripts/ingest.py` | Memakai `pipeline.proses_pdf()` | Logika ingest tidak ditulis dua kali |
| `scripts/tanya.py` | Prompt sitasi per kalimat (dari `prompts.py`); tetap sebagai pembanding RAG dasar | Detail jurnal lain sempat bocor ke sitasi yang salah |
| `requirements.txt` | + `langgraph`, `fastapi`, `uvicorn`, `python-multipart` | Untuk agen dan API |

### Perilaku baru agen
- **Router**: sapaan atau pertanyaan cara pakai dijawab langsung, tanpa koleksi dan tanpa referensi. Kalau ragu, selalu mencari.
- **Grade**: satu panggilan LLM menilai semua kandidat sekaligus (hemat kuota). Hanya chunk relevan yang dikirim ke Generate.
- **Tidak ada chunk relevan** → jawaban standar "belum dibahas di koleksimu", tanpa daftar sumber.
- **Daftar sumber** hanya berisi nomor yang benar-benar disitasi di jawaban.

## Paket 1 — Minggu 1 (20 Nov 2026)
Ingestion PDF ke Qdrant (per `user_id`), `panggil_llm()` dengan cadangan otomatis, tes tanya-jawab dari terminal.
