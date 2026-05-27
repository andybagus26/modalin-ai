# ModalIn AI API

Microservice inference Credit Scoring UMKM — Capstone CC26-PSU259

## Struktur Folder

```
modalin-ai/
├── model/
│   ├── modalin_model.keras     ← hasil training
│   ├── scaler.pkl
│   ├── skor_scaler.pkl
│   ├── label_encoder.pkl
│   └── features.pkl
├── main.py
├── requirements.txt
└── README.md
```

## Setup & Jalankan

```bash
# 1. Buat virtual environment
python -m venv venv

# 2. Aktifkan (Windows)
venv\Scripts\activate
# Aktifkan (Mac/Linux)
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan server
uvicorn main:app --reload --port 8000
```

## Endpoint

| Method | URL       | Deskripsi              |
|--------|-----------|------------------------|
| GET    | /         | Info API               |
| GET    | /health   | Cek status server      |
| POST   | /predict  | Prediksi skor kredit   |
| GET    | /docs     | Swagger UI (otomatis)  |

## Contoh Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "omzet": 5000000,
    "pengeluaran": 3750000,
    "aset": 25000000,
    "hutang": 5000000,
    "freq_trx": 150,
    "lama_bln": 36,
    "jenis_usaha": "Bisnis Kuliner"
  }'
```

## Contoh Response

```json
{
  "skor_kredit": 712.5,
  "status": "Layak",
  "probabilitas": {
    "Tidak Layak": 0.0312,
    "Review": 0.1205,
    "Layak": 0.8483
  },
  "fitur_hitung": {
    "margin_laba": 25.0,
    "dar_ratio": 0.2,
    "oer_ratio": 0.75,
    "avg_trx_value": 33112.58,
    "laba_bersih": 1250000.0,
    "lama_bln": 36.0,
    "freq_trx": 150,
    "hutang": 5000000,
    "jenis_usaha_enc": 0
  },
  "pesan": "Profil keuangan sehat. Layak mendapatkan akses permodalan."
}
```

## Kategori jenis_usaha yang valid

- `Bisnis Kuliner`
- `Jasa & Freelancer`
- `Produk Digital`
- `Produk Kreatif`
- `Toko & E-commerce`
