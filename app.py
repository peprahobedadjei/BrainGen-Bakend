from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth.routes import auth_bp
from predict.routes import predict_bp
from generate.routes import generate_bp
from database.firebase import initialize_firebase
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI()

load_dotenv()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Initialize Firebase using Secret Manager
initialize_firebase()

# Include authentication routes
app.include_router(auth_bp, prefix="/auth")
app.include_router(predict_bp, prefix="/predict")
app.include_router(generate_bp, prefix="/generate")
# Home route
@app.get("/")
async def home():
    return {"message": "Brainwave API is up and running!"}