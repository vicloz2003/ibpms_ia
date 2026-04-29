from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routers.diagram_router import router

load_dotenv()

app = FastAPI(
    title="iBPMS IA Service",
    description="Microservicio de generación de diagramas con IA",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://localhost:*",
        "http://ibpms-frontend.s3-website-us-east-1.amazonaws.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {"message": "iBPMS IA Service running"}
