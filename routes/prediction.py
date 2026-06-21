"""
Prédiction de la demande par zone et heure via Groq.
"""
import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq

router = APIRouter()

# Variable globale _groq supprimée pour éviter le crash au chargement du module.

class PredictionRequest(BaseModel):
    zone: str         # ex: "Akwa", "Bonapriso", "Makepe"
    hour: int         # 0-23
    day_of_week: int  # 0=Lundi, 6=Dimanche
    category: str     # restaurant, boutique, prestataire


class PredictionResponse(BaseModel):
    expected_orders: int
    demand_level: str  # low | medium | high | very_high
    recommended_agents: int
    insight: str


@router.post("/demand", response_model=PredictionResponse)
async def predict_demand(req: PredictionRequest):
    # 1. Sécurisation de la clé d'API
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="La clé GROQ_API_KEY est introuvable dans l'environnement."
        )

    # 2. Instanciation locale du client Groq
    _groq = Groq(api_key=api_key)

    days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    
    # Validation basique de sécurité pour l'index du jour
    if not (0 <= req.day_of_week <= 6):
        raise HTTPException(status_code=400, detail="day_of_week doit être compris entre 0 et 6")

    prompt = f"""
Tu es un analyste de données pour Holla, une plateforme urbaine à Douala, Cameroun.

Prévis la demande pour:
- Zone: {req.zone}
- Heure: {req.hour}h
- Jour: {days[req.day_of_week]}
- Catégorie: {req.category}

Contexte Douala:
- Pics de commande: 12h-14h (déjeuner) et 18h-21h (dîner)
- Samedi et dimanche: +40% vs semaine
- Zones résidentielles (Bonapriso, Bonamoussadi): forte demande le soir
- Zones commerciales (Akwa, Bali): forte demande le midi

Réponds UNIQUEMENT avec un JSON valide:
{{"expected_orders": ..., "demand_level": "...", "recommended_agents": ..., "insight": "..."}}
"""

    response = _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200,
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(content)
        return PredictionResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur de parsing des prédictions de l'IA : {e} | Contenu brut : {content}"
        )