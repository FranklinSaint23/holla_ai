"""
Recommandations intelligentes — personnalisation basée sur l'historique.
"""
import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq

router = APIRouter()


class UserHistory(BaseModel):
    categories_frequentes: list[str]
    heure_commande: int  # 0-23
    ville: str = "Cameroun"
    budget_moyen: int = 5000  # FCFA


class RecommendationRequest(BaseModel):
    user_id: str
    history: UserHistory
    available_partners: list[dict]  # [{id, name, category, rating, avg_price}]


class RecommendationResponse(BaseModel):
    recommended_ids: list[str]
    reasoning: str
    trending_category: str


@router.post("/suggest", response_model=RecommendationResponse)
async def recommend(req: RecommendationRequest):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY manquant")

    _groq = Groq(api_key=api_key)

    heure_label = "matin" if req.history.heure_commande < 12 else ("après-midi" if req.history.heure_commande < 18 else "soir")

    prompt = f"""
Tu es le moteur de recommandation de HOLLA, plateforme de livraison au Cameroun.

Profil utilisateur:
- Catégories préférées: {req.history.categories_frequentes}
- Habitude de commande: {heure_label} (heure={req.history.heure_commande}h)
- Budget moyen: {req.history.budget_moyen} FCFA
- Ville: {req.history.ville}

Partenaires disponibles:
{json.dumps(req.available_partners[:10], indent=2, ensure_ascii=False)}

Recommande les 3 meilleurs partenaires et indique la catégorie tendance.
Réponds UNIQUEMENT en JSON:
{{"recommended_ids": ["id1","id2","id3"], "reasoning": "...", "trending_category": "..."}}
"""

    resp = _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )

    content = resp.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(content)
        return RecommendationResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing error: {e} | Raw: {content}")
