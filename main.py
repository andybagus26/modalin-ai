"""
ModalIn — AI Inference API
FastAPI microservice untuk Credit Scoring UMKM
"""

import os
import pickle
import warnings
import numpy as np
import tensorflow as tf
import keras
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

warnings.filterwarnings("ignore")

# Custom Layer
class CreditScoringLayer(keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units        = units
        self.dropout_rate = dropout_rate
        self.dense        = keras.layers.Dense(units, activation="relu")
        self.bn           = keras.layers.BatchNormalization()
        self.dropout      = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        x = self.bn(x, training=training)
        return self.dropout(x, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "dropout_rate": self.dropout_rate})
        return cfg


# Load semua hasil training 
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model & scaler saat server start."""
    print("⏳ Loading model artifacts...")

    artifacts["model"] = keras.models.load_model(
        os.path.join(MODEL_DIR, "modalin_model.keras"),
        custom_objects={"CreditScoringLayer": CreditScoringLayer},
    )
    with open(os.path.join(MODEL_DIR, "scaler.pkl"),        "rb") as f:
        artifacts["scaler"]       = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "skor_scaler.pkl"),   "rb") as f:
        artifacts["skor_scaler"]  = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        artifacts["le"]           = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "features.pkl"),      "rb") as f:
        artifacts["features"]     = pickle.load(f)

    print("✅ Model loaded!")
    print(f"   Fitur   : {artifacts['features']}")
    print(f"   Kategori: {list(artifacts['le'].classes_)}")
    yield
    artifacts.clear()

# FastAPI
app = FastAPI(
    title       = "ModalIn AI API",
    description = "Microservice Credit Scoring UMKM — Capstone CC26-PSU259",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Ganti dengan domain frontend
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# Schema Input & Output
VALID_JENIS_USAHA = [
    "Bisnis Kuliner",
    "Jasa & Freelancer",
    "Produk Digital",
    "Produk Kreatif",
    "Toko & E-commerce",
]

class UMKMInput(BaseModel):
    omzet        : float = Field(..., gt=0,  description="Rata-rata omzet bulanan (Rp)")
    pengeluaran  : float = Field(..., gt=0,  description="Rata-rata pengeluaran bulanan (Rp)")
    aset         : float = Field(..., gt=0,  description="Estimasi total aset usaha (Rp)")
    hutang       : float = Field(..., ge=0,  description="Total hutang (Rp)")
    freq_trx     : int   = Field(..., gt=0,  description="Frekuensi transaksi digital per bulan")
    lama_bln     : float = Field(..., gt=0,  description="Lama usaha berdiri (bulan)")
    jenis_usaha  : str   = Field(...,        description=f"Kategori usaha: {VALID_JENIS_USAHA}")

    @validator("jenis_usaha")
    def validate_jenis_usaha(cls, v):
        if v not in VALID_JENIS_USAHA:
            raise ValueError(
                f"jenis_usaha harus salah satu dari: {VALID_JENIS_USAHA}"
            )
        return v

    @validator("pengeluaran")
    def validate_pengeluaran(cls, v, values):
        if "omzet" in values and v >= values["omzet"]:
            raise ValueError("pengeluaran tidak boleh lebih besar atau sama dengan omzet")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "omzet"      : 5000000,
                "pengeluaran": 3750000,
                "aset"       : 25000000,
                "hutang"     : 5000000,
                "freq_trx"   : 150,
                "lama_bln"   : 36,
                "jenis_usaha": "Bisnis Kuliner"
            }
        }


class SkorOutput(BaseModel):
    skor_kredit  : float
    status       : str
    probabilitas : dict
    fitur_hitung : dict
    pesan        : str


#hitung fitur dari data mentah
STATUS_MAP  = {0: "Tidak Layak", 1: "Review", 2: "Layak"}
STATUS_MSG  = {
    0: "Profil risiko tinggi. Disarankan perbaiki arus kas terlebih dahulu.",
    1: "Profil perlu ditinjau lebih lanjut. Ada potensi tapi perlu penguatan.",
    2: "Profil keuangan sehat. Layak mendapatkan akses permodalan.",
}

def hitung_fitur(data: UMKMInput) -> dict:
    laba_bersih = data.omzet - data.pengeluaran
    return {
        "margin_laba"   : (laba_bersih / data.omzet) * 100,
        "dar_ratio"     : data.hutang / (data.aset + 1),
        "oer_ratio"     : data.pengeluaran / (data.omzet + 1),
        "avg_trx_value" : data.omzet / (data.freq_trx + 1),
        "laba_bersih"   : laba_bersih,
        "lama_bln"      : data.lama_bln,
        "freq_trx"      : data.freq_trx,
        "hutang"        : data.hutang,
        "jenis_usaha_enc": int(
            artifacts["le"].transform([data.jenis_usaha])[0]
        ),
    }

# Endpoints
@app.get("/", tags=["Info"])
def root():
    return {
        "service" : "ModalIn AI Credit Scoring API",
        "version" : "1.0.0",
        "status"  : "running",
        "endpoint": "/predict",
        "docs"    : "/docs",
    }


@app.get("/health", tags=["Info"])
def health_check():
    model_loaded = "model" in artifacts
    return {
        "status"      : "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=SkorOutput, tags=["Prediksi"])
def predict(data: UMKMInput):
    """
    Prediksi skor kredit UMKM berdasarkan data operasional.

    - **omzet**: rata-rata omzet bulanan dalam Rupiah
    - **pengeluaran**: rata-rata pengeluaran bulanan dalam Rupiah
    - **aset**: estimasi total aset usaha dalam Rupiah
    - **hutang**: total hutang dalam Rupiah (0 jika tidak ada)
    - **freq_trx**: frekuensi transaksi digital per bulan
    - **lama_bln**: lama usaha berdiri dalam bulan
    - **jenis_usaha**: kategori jenis usaha
    """
    try:
        # 1. Hitung fitur
        fitur = hitung_fitur(data)

        # 2. Susun array sesuai urutan FEATURES_FINAL
        row    = np.array(
            [[fitur[f] for f in artifacts["features"]]],
            dtype=np.float32
        )

        # 3. Scale
        row_sc = artifacts["scaler"].transform(row)
        row_t  = tf.cast(row_sc, tf.float32)

        # 4. Prediksi
        skor_norm, status_prob = artifacts["model"](row_t, training=False)

        # 5. Decode hasil
        skor_norm_val = float(skor_norm.numpy()[0][0])
        skor_asli     = float(
            artifacts["skor_scaler"].inverse_transform([[skor_norm_val]])[0][0]
        )
        skor_asli     = max(100.0, min(900.0, skor_asli))  # clamp 100-900
        status_idx    = int(np.argmax(status_prob.numpy()[0]))
        proba_dict    = {
            STATUS_MAP[i]: round(float(p), 4)
            for i, p in enumerate(status_prob.numpy()[0])
        }

        return SkorOutput(
            skor_kredit  = round(skor_asli, 1),
            status       = STATUS_MAP[status_idx],
            probabilitas = proba_dict,
            fitur_hitung = {k: round(v, 4) for k, v in fitur.items()},
            pesan        = STATUS_MSG[status_idx],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
