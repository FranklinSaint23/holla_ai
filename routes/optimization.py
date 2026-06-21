"""
Optimisation de trajets multi-livraisons via Groq.
"""
import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq

router = APIRouter()

# On supprime la variable globale défectueuse _groq ici !

class Stop(BaseModel):
    id: str
    label: str
    lat: float
    lng: float
    type: str  # pickup | delivery


class RouteRequest(BaseModel):
    agent_id: str
    agent_lat: float
    agent_lng: float
    stops: list[Stop]


class RouteResponse(BaseModel):
    optimized_order: list[str]   # ids dans l'ordre optimal
    estimated_total_minutes: int
    reasoning: str


@router.post("/route", response_model=RouteResponse)
async def optimize_route(req: RouteRequest):
    # 1. Sécurité : Vérifier et récupérer la clé API à l'intérieur de la fonction
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="La clé GROQ_API_KEY est introuvable dans l'environnement."
        )

    # 2. Initialiser le client de manière isolée pour cette requête
    _groq = Groq(api_key=api_key)

    prompt = f"""
Tu es un optimiseur de trajets pour livreurs à Douala, Cameroun.

Position du livreur: ({req.agent_lat}, {req.agent_lng})

Arrêts à effectuer:
{json.dumps([s.model_dump() for s in req.stops], indent=2, ensure_ascii=False)}

Règles:
- Un pickup doit toujours précéder sa livraison correspondante
- Minimise la distance totale parcourue
- Tiens compte des embouteillages typiques de Douala (Akwa, Rond-point Deïdo, Carrefour Ndokoti)

Réponds UNIQUEMENT avec un JSON valide:
{{"optimized_order": ["id1","id2",...], "estimated_total_minutes": ..., "reasoning": "..."}}
"""

    response = _groq.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300,
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(content)
        return RouteResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur de parsing du JSON de l'IA : {e} | Contenu brut : {content}"
        )