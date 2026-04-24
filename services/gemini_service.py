import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from models.request_models import DiagramGenerationRequest
from models.response_models import (
    DiagramGenerationResponse, ActivityPartition,
    ActivityNode, ControlFlow, FormSchema, FormField
)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
Eres un experto en modelado de procesos de negocio (BPM).
Tu tarea es generar la estructura de un diagrama de actividades
organizado en swimlanes (calles) basado en una descripción
en lenguaje natural.

Debes responder ÚNICAMENTE con un objeto JSON válido, sin
markdown, sin explicaciones, sin bloques de código.
El JSON debe seguir exactamente esta estructura:

{
  "name": "nombre de la política",
  "description": "descripción breve",
  "partitions": [
    {
      "id": "p1",
      "label": "Nombre del departamento",
      "departmentId": "id_real_del_departamento"
    }
  ],
  "nodes": [
    {
      "id": "n1",
      "label": "Nombre del nodo",
      "partitionId": "p1",
      "type": "INITIAL_NODE|ACTION|DECISION|MERGE|FORK|JOIN|FLOW_FINAL|ACTIVITY_FINAL",
      "formSchema": {
        "fields": [
          {
            "id": "uuid",
            "type": "TEXT|TEXTAREA|NUMBER|DATE|SELECT|FILE|SIGNATURE",
            "label": "Etiqueta del campo",
            "required": true,
            "options": []
          }
        ]
      },
      "metadata": {}
    }
  ],
  "flows": [
    {
      "id": "f1",
      "sourceNodeId": "n1",
      "targetNodeId": "n2",
      "guardCondition": null
    }
  ]
}

Reglas estrictas:
1. SIEMPRE empieza con un nodo INITIAL_NODE
2. SIEMPRE termina con al menos un nodo ACTIVITY_FINAL
3. Cada ACTION debe tener al menos un campo en formSchema
4. Usa SOLO los departamentos proporcionados — asigna cada
   nodo al departamento más lógico según el contexto
5. Si hay bifurcaciones usa DECISION con guardCondition en SpEL:
   ejemplo: #data['campo'] == 'valor'
6. Los IDs deben ser strings únicos cortos: p1, p2, n1, n2, f1...
7. Responde SOLO con el JSON, nada más
8. Las opciones de campos SELECT deben ser strings simples,
   no objetos. Ejemplo correcto: ["Aprobado", "Rechazado"]
   Ejemplo incorrecto: [{"value": "aprobado", "label": "Aprobado"}]
"""


def build_user_prompt(request: DiagramGenerationRequest) -> str:
    dept_list = "\n".join([
        f"- ID: {d.id}, Nombre: {d.name}"
        for d in request.departments
    ])
    return f"""
Departamentos disponibles en la empresa:
{dept_list}

Nombre de la política: {request.policyName or 'Sin nombre'}

Descripción del proceso:
{request.description}

Genera el diagrama JSON siguiendo las reglas indicadas.
Usa los IDs de departamento exactos proporcionados arriba.
"""


def normalize_options(data: dict) -> dict:
    for node in data.get("nodes", []):
        form_schema = node.get("formSchema", {})
        for field in form_schema.get("fields", []):
            options = field.get("options", [])
            normalized = []
            for opt in options:
                if isinstance(opt, dict):
                    # Usar 'value' para preservar semántica de negocio
                    # fallback a 'label' si no hay value
                    normalized.append(
                        opt.get("value") or opt.get("label") or str(opt)
                    )
                else:
                    normalized.append(str(opt))
            field["options"] = normalized
    return data


async def generate_diagram(
    request: DiagramGenerationRequest
) -> DiagramGenerationResponse:
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT
    )

    user_prompt = build_user_prompt(request)

    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=4096,
        )
    )

    raw_text = response.text.strip()

    # Limpiar posibles bloques markdown
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(lines[1:-1])

    data = json.loads(raw_text)
    data = normalize_options(data)

    return DiagramGenerationResponse(**data)
