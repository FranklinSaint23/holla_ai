from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routes.assignment import router as assignment_router
from routes.optimization import router as optimization_router
from routes.prediction import router as prediction_router
from routes.chatbot import router as chatbot_router
from routes.recommendations import router as recommendations_router
from routes.price_estimate import router as price_router

load_dotenv()

app = FastAPI(
    title="Holla AI Microservice",
    description="Chatbot IA, affectation livreurs, optimisation trajets, prédiction demande, recommandations, estimation prix",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assignment_router,      prefix="/ai/assignment",      tags=["Assignment"])
app.include_router(optimization_router,   prefix="/ai/optimization",   tags=["Optimization"])
app.include_router(prediction_router,     prefix="/ai/prediction",     tags=["Prediction"])
app.include_router(chatbot_router,        prefix="/ai/chatbot",        tags=["Chatbot"])
app.include_router(recommendations_router,prefix="/ai/recommendations",tags=["Recommendations"])
app.include_router(price_router,          prefix="/ai/pricing",        tags=["Pricing"])

@app.get("/health")
def health():
    return {"status": "ok", "service": "holla-ai"}
