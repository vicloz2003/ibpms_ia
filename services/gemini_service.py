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


def normalize_diagram_data(data: dict) -> dict:
    partitions = data.get("partitions", [])
    default_partition_id = partitions[0]["id"] if partitions else "p1"

    for node in data.get("nodes", []):
        # Normalizar partitionId nulo
        if not node.get("partitionId"):
            node["partitionId"] = default_partition_id

        # Normalizar formSchema
        form_schema = node.get("formSchema", {})
        if form_schema is None:
            node["formSchema"] = {"fields": []}
            form_schema = node["formSchema"]

        for field in form_schema.get("fields", []):
            options = field.get("options", [])
            normalized = []
            for opt in options:
                if isinstance(opt, dict):
                    normalized.append(
                        opt.get("value") or opt.get("label") or str(opt)
                    )
                else:
                    normalized.append(str(opt))
            field["options"] = normalized

    return data


def _xe(text: str) -> str:
    """Escape special XML characters in attribute values."""
    return (text
            .replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def generate_bpmn_xml(data: dict) -> str:
    partitions = data.get("partitions", [])
    nodes = data.get("nodes", [])
    flows = data.get("flows", [])

    # Layout constants
    PARTICIPANT_X = 130
    PARTICIPANT_Y = 80
    LANE_LABEL_W = 30        # bpmn-js pool header width
    LANE_INNER_X = PARTICIPANT_X + LANE_LABEL_W  # x where lanes start = 160
    COL_W = 200              # horizontal spacing between nodes
    LANE_H = 200             # height of each swimlane
    NODE_START_X = LANE_INNER_X + 70  # first node column, past lane label

    # Calculate total width based on widest lane
    max_nodes_in_lane = max(
        (len([n for n in nodes if n.get("partitionId") == p.get("id")]) for p in partitions),
        default=1
    )
    total_width = max(900, NODE_START_X - PARTICIPANT_X + max_nodes_in_lane * COL_W + 80)
    lane_width = total_width - LANE_LABEL_W
    total_height = max(LANE_H, len(partitions) * LANE_H)

    # ── Process elements (semantic BPMN) ──────────────────────────────────────
    process_elements = ""
    for node in nodes:
        ntype = node.get("type", "ACTION")
        nid = node.get("id", "")
        nlabel = _xe(node.get("label", ""))
        if ntype == "INITIAL_NODE":
            process_elements += f'<bpmn:startEvent id="{nid}" name="{nlabel}"/>\n'
        elif ntype == "ACTIVITY_FINAL":
            process_elements += f'<bpmn:endEvent id="{nid}" name="{nlabel}"/>\n'
        elif ntype in ("DECISION", "MERGE"):
            process_elements += f'<bpmn:exclusiveGateway id="{nid}" name="{nlabel}"/>\n'
        elif ntype in ("FORK", "JOIN"):
            process_elements += f'<bpmn:parallelGateway id="{nid}" name="{nlabel}"/>\n'
        else:
            process_elements += f'<bpmn:task id="{nid}" name="{nlabel}"/>\n'

    for flow in flows:
        fid = flow.get("id", "")
        src = flow.get("sourceNodeId", "")
        tgt = flow.get("targetNodeId", "")
        cond = flow.get("guardCondition")
        if cond:
            process_elements += (
                f'<bpmn:sequenceFlow id="{fid}" sourceRef="{src}" targetRef="{tgt}">'
                f'<bpmn:conditionExpression>{_xe(cond)}</bpmn:conditionExpression>'
                f'</bpmn:sequenceFlow>\n'
            )
        else:
            process_elements += (
                f'<bpmn:sequenceFlow id="{fid}" sourceRef="{src}" targetRef="{tgt}"/>\n'
            )

    # ── Lane semantic elements ────────────────────────────────────────────────
    lane_elements = ""
    for partition in partitions:
        pid = partition.get("id", "")
        plabel = _xe(partition.get("label", ""))
        refs = "".join(
            f'<bpmn:flowNodeRef>{n["id"]}</bpmn:flowNodeRef>'
            for n in nodes if n.get("partitionId") == pid
        )
        lane_elements += f'<bpmn:lane id="{pid}" name="{plabel}">{refs}</bpmn:lane>\n'

    # ── DI: participant shape ─────────────────────────────────────────────────
    participant_shape = (
        f'<bpmndi:BPMNShape id="Participant_1_di" bpmnElement="Participant_1" isHorizontal="true">\n'
        f'  <dc:Bounds x="{PARTICIPANT_X}" y="{PARTICIPANT_Y}" '
        f'width="{total_width}" height="{total_height}"/>\n'
        f'</bpmndi:BPMNShape>\n'
    )

    # ── DI: lane shapes (CRITICAL — bpmn-js won't draw borders without these) ─
    lane_shapes = ""
    for idx, partition in enumerate(partitions):
        pid = partition.get("id", "")
        lane_y = PARTICIPANT_Y + idx * LANE_H
        lane_shapes += (
            f'<bpmndi:BPMNShape id="{pid}_di" bpmnElement="{pid}" isHorizontal="true">\n'
            f'  <dc:Bounds x="{LANE_INNER_X}" y="{lane_y}" '
            f'width="{lane_width}" height="{LANE_H}"/>\n'
            f'</bpmndi:BPMNShape>\n'
        )

    # ── DI: node shapes ───────────────────────────────────────────────────────
    node_shapes = ""
    for node in nodes:
        ntype = node.get("type", "ACTION")
        nid = node.get("id", "")
        pid = node.get("partitionId", "")
        partition_idx = next(
            (j for j, p in enumerate(partitions) if p.get("id") == pid), 0
        )
        lane_nodes = [n for n in nodes if n.get("partitionId") == pid]
        col_idx = lane_nodes.index(node) if node in lane_nodes else 0

        if ntype in ("INITIAL_NODE", "ACTIVITY_FINAL"):
            w, h = 36, 36
        elif ntype in ("DECISION", "MERGE", "FORK", "JOIN"):
            w, h = 50, 50
        else:
            w, h = 100, 80

        nx = NODE_START_X + col_idx * COL_W
        # Center the node vertically in its lane
        lane_y = PARTICIPANT_Y + partition_idx * LANE_H
        ny = lane_y + (LANE_H - h) // 2

        node_shapes += (
            f'<bpmndi:BPMNShape id="{nid}_di" bpmnElement="{nid}">\n'
            f'  <dc:Bounds x="{nx}" y="{ny}" width="{w}" height="{h}"/>\n'
            f'</bpmndi:BPMNShape>\n'
        )

    # ── DI: edges ─────────────────────────────────────────────────────────────
    edges = ""
    for flow in flows:
        fid = flow.get("id", "")
        edges += f'<bpmndi:BPMNEdge id="{fid}_di" bpmnElement="{fid}"/>\n'

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" '
        'xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI" '
        'xmlns:dc="http://www.omg.org/spec/DD/20100524/DC" '
        'id="Definitions_ia" targetNamespace="http://bpmn.io/schema/bpmn">\n'
        '  <bpmn:collaboration id="Collaboration_1">\n'
        '    <bpmn:participant id="Participant_1" name="Proceso" processRef="Process_1"/>\n'
        '  </bpmn:collaboration>\n'
        '  <bpmn:process id="Process_1" isExecutable="false">\n'
        '    <bpmn:laneSet id="LaneSet_1">\n'
        f'{lane_elements}'
        '    </bpmn:laneSet>\n'
        f'{process_elements}'
        '  </bpmn:process>\n'
        '  <bpmndi:BPMNDiagram id="BPMNDiagram_1">\n'
        '    <bpmndi:BPMNPlane id="BPMNPlane_1" bpmnElement="Collaboration_1">\n'
        f'{participant_shape}'
        f'{lane_shapes}'
        f'{node_shapes}'
        f'{edges}'
        '    </bpmndi:BPMNPlane>\n'
        '  </bpmndi:BPMNDiagram>\n'
        '</bpmn:definitions>'
    )
    return xml


async def generate_diagram(
    request: DiagramGenerationRequest
) -> DiagramGenerationResponse:
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT
        )

        user_prompt = build_user_prompt(request)

        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=8192,
            )
        )

        raw_text = response.text.strip()
        print(f"[GEMINI] Raw response: {raw_text[:200]}")

        # Limpiar posibles bloques markdown
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            raw_text = "\n".join(lines[1:-1])

        data = json.loads(raw_text)
        data = normalize_diagram_data(data)
        bpmn_xml = generate_bpmn_xml(data)
        response_data = DiagramGenerationResponse(**data)
        response_data.suggestedBpmnXml = bpmn_xml
        return response_data
    except Exception:
        import traceback
        print(f"[GEMINI] Error: {traceback.format_exc()}")
        raise
