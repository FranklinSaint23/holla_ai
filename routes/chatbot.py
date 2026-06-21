"""
Chatbot IA HOLLA — assistant conversationnel pour les utilisateurs.
"""
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq

router = APIRouter()

_SYSTEM_PROMPT = """Tu es Holla AI, l'assistant intelligent de la plateforme HOLLA.
HOLLA est une super-application de livraison et services au Cameroun (couverture nationale).
Tu aides les utilisateurs à :
- Commander de la nourriture et des biens
- Trouver des prestataires (plombiers, électriciens, etc.)
- Suivre leurs livraisons
- Comprendre les tarifs et frais
- Résoudre les problèmes avec leurs commandes
- Payer via MTN Mobile Money ou Orange Money
- Comprendre comment fonctionner sur la plateforme

Règles :
1. Réponds toujours en français sauf si l'utilisateur écrit en anglais
2. Sois concis, chaleureux et professionnel
3. Si tu ne peux pas résoudre un problème, suggère de contacter le support
4. Ne mentionne jamais une ville spécifique comme unique zone — HOLLA couvre TOUT le Cameroun
5. Pour les litiges, dis toujours à l'utilisateur d'aller dans Historique → Détails → Signaler
6. Maximum 3 phrases par réponse, sauf si l'utilisateur demande plus de détails

Suggestions rapides à proposer à la fin de tes réponses (si pertinent) :
- "Comment passer une commande ?"
- "Comment suivre ma livraison ?"
- "Modes de paiement acceptés ?"
- "Comment devenir livreur ?"
- "Contacter le support"
"""


class ChatMessage(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    user_role: str = "client"
    language: str = "fr"


class ChatResponse(BaseModel):
    reply: str
    suggestions: list[str]


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY manquant")

    _groq = Groq(api_key=api_key)

    # Construction du contexte rôle
    role_ctx = {
        "client": "L'utilisateur est un CLIENT qui commande des biens/services.",
        "delivery": "L'utilisateur est un LIVREUR partenaire de HOLLA.",
        "partner": "L'utilisateur est un PARTENAIRE (restaurant/boutique) sur HOLLA.",
        "provider": "L'utilisateur est un PRESTATAIRE de services (plombier, électricien, etc.).",
        "admin": "L'utilisateur est un ADMINISTRATEUR de la plateforme.",
    }
    role_info = role_ctx.get(req.user_role, role_ctx["client"])
    system = f"{_SYSTEM_PROMPT}\n\nContexte actuel : {role_info}"

    messages = [{"role": "system", "content": system}]
    for m in req.messages[-10:]:  # garde max 10 messages pour le contexte
        messages.append({"role": m.role, "content": m.content})

    try:
        resp = _groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        reply = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Suggestions contextuelles basées sur le dernier message user
    last_msg = req.messages[-1].content.lower() if req.messages else ""
    suggestions = _get_suggestions(last_msg, req.user_role)

    return ChatResponse(reply=reply, suggestions=suggestions)


def _get_suggestions(msg: str, role: str) -> list[str]:
    if "commande" in msg or "commander" in msg:
        return ["Comment payer ?", "Frais de livraison ?", "Suivi en temps réel ?"]
    if "livraison" in msg or "livreur" in msg:
        return ["Temps estimé ?", "Contacter le livreur", "Signaler un problème"]
    if "paiement" in msg or "payer" in msg:
        return ["MTN Mobile Money", "Orange Money", "Paiement en espèces ?"]
    if "prestataire" in msg or "plombier" in msg or "électricien" in msg:
        return ["Demander un devis", "Voir les avis", "Disponibilité immédiate ?"]
    if role == "delivery":
        return ["Accepter une commande", "Mes revenus aujourd'hui", "Zone de livraison"]
    if role == "partner":
        return ["Gérer mon menu", "Nouvelles commandes", "Mes statistiques"]
    return ["Passer une commande", "Trouver un prestataire", "Mode de paiement ?"]
