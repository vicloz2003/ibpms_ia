# ibpms-ia

## Instalación
pip install -r requirements.txt

## Configuración
Copia .env.example a .env y agrega tu GEMINI_API_KEY

## Ejecutar
uvicorn main:app --reload --port 8000

## Endpoint principal
POST http://localhost:8000/api/ia/generate-diagram
