# Asisten Riset & Skripsi (Agentic RAG)

Asisten yang menjawab pertanyaan riset berdasarkan jurnal PDF milik pengguna, dengan sitasi sampai nomor halaman.

Status: **Minggu 3** — agen LangGraph (router → retrieve → grade → generate) + API FastAPI.
Lihat `CHANGELOG.md` untuk daftar perubahan.

## Menjalankan agen (minggu 3)

```bash
pip install -r requirements.txt                          # tambahan: langgraph, fastapi, uvicorn
python -m scripts.tanya_agen "Apa perbedaan Self-RAG dan CRAG?" --user muhammad
python -m eval.run_eval --user muhammad                  # jalankan semua soal uji
uvicorn app.main:app --reload                            # API, coba di http://localhost:8000/docs
```

`scripts.tanya` (RAG dasar tanpa agen) tetap ada sebagai pembanding.

## Persiapan (sekali saja)

Butuh: Python 3.11+, Docker Desktop.

```bash
# 1. Virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install library
pip install -r requirements.txt

# 3. Isi API key
copy .env.example .env      # Windows  (macOS/Linux: cp .env.example .env)
#    lalu buka .env dan isi GROQ_API_KEY (gratis: https://console.groq.com/keys)
#    GEMINI_API_KEY / MISTRAL_API_KEY opsional sebagai cadangan

# 4. Jalankan Qdrant
docker compose up -d
#    cek dashboard: http://localhost:6333/dashboard
```

## Langkah minggu 1

```bash
# a. Tes koneksi LLM (ringan & berat)
python -m app.llm

# b. Cek satu PDF terbaca rapi atau tidak
python -m app.ingestion.pdf_parser data/papers/contoh.pdf

# c. Masukkan semua PDF di folder ke Qdrant
python -m scripts.ingest data/papers --user rina

# d. Tes tanya-jawab dengan sitasi
python -m scripts.tanya "Metode apa yang paling sering dipakai untuk deteksi hoaks?" --user rina
```

Saat pertama kali dijalankan, model embedding (`multilingual-e5-small`, sekitar 470 MB) diunduh otomatis lalu disimpan di cache.

## Struktur

```
app/
  config.py              # semua pengaturan dari .env
  llm.py                 # panggil_llm(): Groq + cadangan Gemini/Mistral via litellm
  query.py               # multi-query EN/ID + sub-query untuk perbandingan
  main.py                # API FastAPI: /upload, /chat
  agent/
    state.py             # ResearchState
    prompts.py           # semua prompt
    nodes.py             # router, retrieve, grade, generate, jawab_langsung, tidak_ada
    graph.py             # graf LangGraph + tanya_agen()
  ingestion/
    pdf_parser.py        # teks per halaman + tebakan cepat metadata
    metadata.py          # judul/penulis/tahun dari halaman 1 (LLM ringan)
    chunker.py           # potong per paragraf, tidak melewati batas halaman
    indexer.py           # embedding + simpan/cari di Qdrant (difilter per user_id)
    pipeline.py          # proses_pdf(): ingest satu PDF dari awal sampai akhir
scripts/
  ingest.py              # CLI: masukkan folder PDF
  tanya.py               # CLI: RAG dasar (pembanding)
  tanya_agen.py          # CLI: lewat agen, dengan jejak node
eval/
  questions.jsonl        # soal uji
  run_eval.py            # jalankan & skor soal uji
data/papers/             # taruh PDF di sini (tidak ikut di-commit)
```

## Catatan desain

- **Per user**: setiap chunk menyimpan `user_id` dan setiap pencarian difilter dengan itu.
- **Satu pintu LLM**: semua kode memanggil `panggil_llm(prompt, peran="ringan"|"berat")`.
  Ganti penyedia cukup dari `.env`.
- **Metadata adalah tebakan**: judul/penulis/tahun diambil dari metadata PDF atau halaman 1.
  Kolom yang perlu dicek ditampilkan saat ingest. Nanti di UI pengguna bisa mengoreksinya.
- **PDF hasil scan dilewati** (belum ada OCR).

## Berikutnya (minggu 2–3)

- Evaluasi 10 pertanyaan dan perbaiki prompt sitasi
- `agent/state.py` + `agent/graph.py`: router → retrieve → grade → generate dengan LangGraph
- Endpoint FastAPI `/upload` dan `/chat`
