import os
import cv2
import numpy as np
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="Banana Expert AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# ✅ 1. CONFIG & CONNECTIONS
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Supabase disabled for deployment compatibility
# from supabase import create_client, Client
# SUPABASE_URL = "https://ypdmdfdwzldsifijajrm.supabase.co"
# SUPABASE_KEY = "eyJhbG...U8EE"
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------
# ✅ 2. LOAD MODELS
# -------------------------
# 🔥 โมเดลหลัก (จำแนกสายพันธุ์) - ใช้ ONNX
try:
    MODEL_REAL = YOLO(os.path.join(MODEL_DIR, "best_modelv8sbg.onnx"))
except:
    MODEL_REAL = YOLO(os.path.join(MODEL_DIR, "best_modelv8nbg.onnx"))

# 🔥 โมเดลกรอง (มี/ไม่มีกล้วย) - ใช้ ONNX
try:
    MODEL_FILTER = YOLO(os.path.join(MODEL_DIR, "best_m1_bgv8s.onnx"))
except:
    MODEL_FILTER = None

CLASS_KEYS = {
    0: "candyapple", 1: "namwa", 2: "namwadam", 3: "homthong",
    4: "nak", 5: "thepphanom", 6: "kai", 7: "lepchanggud",
    8: "ngachang", 9: "huamao",
}

async def preprocess_image(file: UploadFile):
    try:
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (640, 640))
        return img
    except:
        return None

# -------------------------
# ✅ 3. API ROUTES
# -------------------------

@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    allow_storage: str = Form("false")
):
    try:
        # 1. preprocess
        img = await preprocess_image(image)
        if img is None:
            return {"success": False, "reason": "invalid_image_format"}

        # -------------------------
        # ✅ 2. FILTER STAGE (NEW)
        # -------------------------
        if MODEL_FILTER is not None:
            filter_results = MODEL_FILTER(img, conf=0.1, iou=0.45, verbose=False)[0]

            if not filter_results.boxes or len(filter_results.boxes) == 0:
                return {"success": False, "reason": "no_banana_detected"}

        # -------------------------
        # ✅ 3. MAIN MODEL (เดิม)
        # -------------------------
        results = MODEL_REAL(img, conf=0.80, iou=0.45, augment=False, verbose=False)[0]

        if not results.boxes or len(results.boxes) == 0:
            return {"success": False, "reason": "no_banana_detected"}

        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)
        best_idx = int(confs.argmax())

        final_conf = float(confs[best_idx])
        banana_key = CLASS_KEYS.get(int(clses[best_idx]), "unknown")

        # -------------------------
        # ✅ 4. SAVE TO SUPABASE (disabled for now)
        # -------------------------
        # Storage feature temporarily disabled due to deployment compatibility
        # Uncomment when supabase dependency is resolved
        """
        if allow_storage.lower() == "true":
            await image.seek(0)
            file_bits = await image.read()
            file_path = f"public/{int(time.time())}_{image.filename}"

            supabase.storage.from_("banana_images").upload(
                path=file_path,
                file=file_bits,
                file_options={"content-type": image.content_type}
            )

            image_url = supabase.storage.from_("banana_images").get_public_url(file_path)

            supabase.table("detection_logs").insert({
                "image_url": image_url,
                "predicted_variety": banana_key,
                "confidence": round(final_conf, 3),
                "is_consent": True
            }).execute()

            print("✨ บันทึกข้อมูลลง Supabase เรียบร้อย!")
        """

        return {
            "success": True,
            "banana_key": banana_key,
            "confidence": round(final_conf, 3)
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "reason": "server_error", "detail": str(e)}

@app.get("/")
async def root():
    return {"message": "Banana AI Server is running!", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)