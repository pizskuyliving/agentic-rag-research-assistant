"""Satu-satunya pintu ke LLM.

Semua node agen (router, grade, generate, ...) memanggil `panggil_llm`.
Kalau nanti ingin pindah penyedia (misalnya ke Claude API), cukup ubah .env.

Peran:
- "ringan": tugas pendek & sering (router, grade, query) -> openai/gpt-oss-20b
- "berat" : menyusun jawaban & self-check            -> openai/gpt-oss-120b

Catatan model gpt-oss: ini model *reasoning*. Ia "berpikir" dulu, dan token berpikir
ikut dihitung dalam max_tokens. Karena itu:
- reasoning_effort diset "low" (cukup untuk tugas klasifikasi & sitasi, hemat token)
- max_tokens tidak boleh terlalu kecil, kalau tidak jawaban bisa kosong

Jika kena batas kuota per menit (429), panggilan menunggu sesuai saran server lalu
mencoba lagi. Jika tetap gagal, litellm pindah ke cadangan (Gemini/Mistral) bila diisi.
"""
import re
import time

import litellm
from litellm import Router

from app import config

# Jika model cadangan tidak mendukung suatu parameter (mis. reasoning_effort),
# parameter itu dibuang otomatis alih-alih membuat panggilan gagal.
litellm.drop_params = True

# Batas bawah max_tokens: ruang untuk "berpikir" + jawaban
MIN_MAX_TOKENS = 600
MAKS_COBA_ULANG = 3


def _bangun_router() -> Router:
    model_list = [
        {"model_name": "ringan", "litellm_params": {"model": config.MODEL_RINGAN, "api_key": config.GROQ_API_KEY}},
        {"model_name": "berat", "litellm_params": {"model": config.MODEL_BERAT, "api_key": config.GROQ_API_KEY}},
    ]
    cadangan = []
    if config.GEMINI_API_KEY:
        model_list.append({"model_name": "cadangan-gemini",
                           "litellm_params": {"model": config.MODEL_CADANGAN_GEMINI, "api_key": config.GEMINI_API_KEY}})
        cadangan.append("cadangan-gemini")
    if config.MISTRAL_API_KEY:
        model_list.append({"model_name": "cadangan-mistral",
                           "litellm_params": {"model": config.MODEL_CADANGAN_MISTRAL, "api_key": config.MISTRAL_API_KEY}})
        cadangan.append("cadangan-mistral")

    # Kuota Groq dihitung per model, jadi model Groq lain bisa jadi cadangan.
    fallbacks = [
        {"ringan": ["berat", *cadangan]},
        {"berat": [*cadangan, "ringan"]},
    ]
    # num_retries=0: penantian saat 429 ditangani sendiri di panggil_llm (lebih sabar)
    return Router(model_list=model_list, fallbacks=fallbacks, num_retries=0)


_router = None


def _detik_tunggu(err: Exception) -> float | None:
    """Ambil saran tunggu dari pesan 429, mis. 'Please try again in 12.945s'."""
    teks = str(err)
    if "rate_limit" not in teks and "429" not in teks and "RateLimit" not in type(err).__name__:
        return None
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", teks)
    if not m:
        return 15.0
    return int(m.group(1) or 0) * 60 + float(m.group(2)) + 1.0


def _completion(peran: str, **kwargs):
    global _router
    if _router is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY belum diisi di file .env")
        _router = _bangun_router()

    for percobaan in range(MAKS_COBA_ULANG + 1):
        try:
            return _router.completion(model=peran, **kwargs)
        except Exception as e:
            tunggu = _detik_tunggu(e)
            if tunggu is None or percobaan == MAKS_COBA_ULANG:
                raise
            print(f"   (kuota per menit penuh, menunggu {tunggu:.0f} detik...)")
            time.sleep(min(tunggu, 65))


def panggil_llm(prompt: str, peran: str = "berat", system: str | None = None,
                temperature: float = 0.0, max_tokens: int = 1024, json_mode: bool = False) -> str:
    """Kirim prompt ke LLM sesuai peran ("ringan" / "berat"), kembalikan teks jawabannya.

    json_mode=True memaksa model membalas JSON valid (prompt harus menyebut kata "JSON").
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    extra = {"reasoning_effort": config.REASONING_EFFORT}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}

    resp = _completion(peran, messages=messages, temperature=temperature,
                       max_tokens=max(max_tokens, MIN_MAX_TOKENS), **extra)
    return resp.choices[0].message.content or ""


if __name__ == "__main__":
    # Tes cepat: python -m app.llm   (menampilkan model yang BENAR-BENAR menjawab)
    for peran, tanya in [("ringan", "Jawab satu kata: ibu kota Riau?"),
                         ("berat", "Jelaskan RAG dalam satu kalimat bahasa Indonesia.")]:
        r = _completion(peran, messages=[{"role": "user", "content": tanya}],
                        max_tokens=MIN_MAX_TOKENS, reasoning_effort=config.REASONING_EFFORT)
        print(f"{peran:6} [{r.model}]: {r.choices[0].message.content}")
