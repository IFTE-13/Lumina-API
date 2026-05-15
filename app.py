# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import tempfile
import os
import pandas as pd
import traceback
from feature_extractor import extract_features

app = FastAPI(title="Lumina — Malware Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model artifacts ───────────────────────────────────
model = scaler = feature_columns = None
try:
    model           = joblib.load("malware_model.pkl")
    scaler          = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    print(f"Model loaded — {len(feature_columns)} features expected")
except FileNotFoundError:
    print("Model files not found. Run: python train_model.py")
except Exception as e:
    print(f"Error loading model: {e}")


@app.get("/")
async def root():
    return {"message": "Lumina Malware Detection API", "model_loaded": model is not None}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_ready": model is not None}

@app.post("/predict")
async def predict_malware(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".exe"):
        raise HTTPException(status_code=400, detail="Only .exe files are accepted")

    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".exe") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        features_dict = extract_features(tmp_path)

        if not features_dict:
            raise HTTPException(
                status_code=422,
                detail="Could not extract PE features. File may be corrupt or not a valid PE."
            )

        # ── Align to training columns ──────────────────────
        # Start with a zero-filled row for ALL training columns
        final_features = pd.DataFrame(
            [[0] * len(feature_columns)],
            columns=feature_columns
        )

        # Fill in only the columns that exist in BOTH
        # extractor output AND training feature set
        matched  = 0
        missing  = []
        for col in feature_columns:
            if col in features_dict:
                final_features[col] = features_dict[col]
                matched += 1
            else:
                missing.append(col)

        if missing:
            print(f"  Columns missing from extractor "
                  f"(zero-filled): {missing}")

        overlap_pct = matched / len(feature_columns) * 100
        print(f"  Feature overlap: {matched}/{len(feature_columns)} "
              f"({overlap_pct:.1f}%)")

        # Warn if overlap is too low — predictions will be unreliable
        if overlap_pct < 80:
            print(f"  WARNING: Low overlap ({overlap_pct:.1f}%) — "
                  f"retrain model with updated feature_extractor.py")

        final_features = final_features.fillna(0)

        # ── Scale and predict ──────────────────────────────
        features_scaled = pd.DataFrame(
            scaler.transform(final_features),
            columns=feature_columns
        )

        prediction  = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        verdict    = "MALICIOUS" if prediction == 1 else "BENIGN"
        confidence = round(
            float(probability[1] if prediction == 1 else probability[0]) * 100,
            2
        )

        print(f"  Result: {verdict} ({confidence}%)")

        return {
            "filename":              file.filename,
            "verdict":               verdict,
            "confidence":            confidence,
            "probability_benign":    round(float(probability[0]), 4),
            "probability_malicious": round(float(probability[1]), 4),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {str(e)}"
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)