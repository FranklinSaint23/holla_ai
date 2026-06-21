"""
Estimation intelligente des prix de livraison et services.
"""
import os
import json
import math
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq

router = APIRouter()


class PriceEstimateRequest(BaseModel):
    service_type: str  # 'delivery' | 'plumber' | 'electrician' | 'cleaning' | 'taxi'
    pickup_lat: float = 0
    pickup_lng: float = 0
    delivery_lat: float = 0
    delivery_lng: float = 0
    description: str = ""
    heure: int = 12  # heure de la demande
    is_urgent: bool = False


class PriceEstimateResponse(BaseModel):
    min_price: int
    max_price: int
    estimated_duration_min: int
    currency: str = "XAF"
    breakdown: dict


def _distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return R * 2 * math.asin(math.sqrt(a))


@router.post("/estimate", response_model=PriceEstimateResponse)
async def estimate_price(req: PriceEstimateRequest):
    # Calcul de base sans IA (rapide et fiable)
    dist_km = _distance_km(req.pickup_lat, req.pickup_lng, req.delivery_lat, req.delivery_lng)

    base_prices = {
        "delivery":     {"base": 500, "per_km": 150, "min": 500,   "max_mult": 2.5},
        "plumber":      {"base": 5000, "per_km": 0,  "min": 5000,  "max_mult": 3.0},
        "electrician":  {"base": 4000, "per_km": 0,  "min": 4000,  "max_mult": 3.0},
        "cleaning":     {"base": 8000, "per_km": 0,  "min": 8000,  "max_mult": 2.0},
        "taxi":         {"base": 500,  "per_km": 200, "min": 500,   "max_mult": 2.0},
    }
    cfg = base_prices.get(req.service_type, base_prices["delivery"])

    base = cfg["base"] + int(dist_km * cfg["per_km"])
    urgency_mult = 1.5 if req.is_urgent else 1.0
    peak_mult = 1.2 if req.heure in range(7, 9) or req.heure in range(17, 20) else 1.0

    min_price = max(cfg["min"], int(base * urgency_mult * peak_mult))
    max_price = int(min_price * cfg["max_mult"])

    est_duration = max(10, int(dist_km * 3 + 5)) if req.service_type in ("delivery", "taxi") else 30

    breakdown = {
        "base": cfg["base"],
        "distance_km": round(dist_km, 1),
        "distance_fee": int(dist_km * cfg["per_km"]),
        "urgency_surcharge": int(base * (urgency_mult - 1)),
        "peak_hour_surcharge": int(base * urgency_mult * (peak_mult - 1)),
    }

    # Enrichissement IA si description fournie
    if req.description:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            _groq = Groq(api_key=api_key)
            prompt = f"""
Service: {req.service_type}, description: "{req.description}"
Estimation initiale: {min_price}–{max_price} FCFA
Ajuste le prix si la description indique une complexité particulière.
Réponds UNIQUEMENT en JSON: {{"adjusted_min": ..., "adjusted_max": ..., "note": "..."}}
"""
            try:
                resp = _groq.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=100,
                )
                content = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
                data = json.loads(content)
                min_price = data.get("adjusted_min", min_price)
                max_price = data.get("adjusted_max", max_price)
                breakdown["ai_note"] = data.get("note", "")
            except Exception:
                pass  # garde l'estimation de base si l'IA échoue

    return PriceEstimateResponse(
        min_price=min_price,
        max_price=max_price,
        estimated_duration_min=est_duration,
        breakdown=breakdown,
    )
