"""
Affectation intelligente des livreurs aux commandes via Groq.
"""
import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq

router = APIRouter()

# On ne crée plus l'instance ici globalement pour éviter le crash au démarrage !

class Order(BaseModel):
    order_id: str
    pickup_lat: float
    pickup_lng: float
    delivery_lat: float
    delivery_lng: float
    priority: str = "normal"   # normal | urgent


class Agent(BaseModel):
    agent_id: str
    name: str
    current_lat: float
    current_lng: float
    vehicle: str
    rating: float
    active_deliveries: int = 0


class AssignmentRequest(BaseModel):
    order: Order
    available_agents: list[Agent]


class AssignmentResponse(BaseModel):
    assigned_agent_id: str
    reason: str
    estimated_minutes: int


@router.post("/assign", response_model=AssignmentResponse)
async def assign_delivery(req: AssignmentRequest):
    if not req.available_agents:
        raise HTTPException(status_code=404, detail="Aucun livreur disponible")

    # 1. On récupère la clé à l'intérieur de la fonction (le .env sera alors chargé à 100%)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="La clé GROQ_API_KEY est manquante dans l'environnement.")

    # 2. On instancie le client Groq de manière sécurisée
    _groq = Groq(api_key=api_key)

    prompt = f"""
Tu es un système d'affectation de livreurs pour Holla, une plateforme de livraison à Douala, Cameroun.

Commande:
- ID: {req.order.order_id}
- Pickup: ({req.order.pickup_lat}, {req.order.pickup_lng})
- Livraison: ({req.order.delivery_lat}, {req.order.delivery_lng})
- Priorité: {req.order.priority}

Livreurs disponibles:
{json.dumps([a.model_dump() for a in req.available_agents], indent=2, ensure_ascii=False)}

Critères de choix (par ordre d'importance):
1. Proximité du pickup (distance euclidienne)
2. Note (rating) du livreur
3. Nombre de livraisons actives (moins c'est mieux)
4. Type de véhicule (moto > bicyclette pour les longues distances)

Réponds UNIQUEMENT avec un JSON valide:
{{"assigned_agent_id": "...", "reason": "...", "estimated_minutes": ...}}
"""

    response = _groq.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200,
    )

    content = response.choices[0].message.content.strip()
    # Nettoyer les backticks si présents
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(content)
        return AssignmentResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing error: {e} | Raw: {content}")