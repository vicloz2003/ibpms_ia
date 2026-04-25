import os
import json
from collections import deque, defaultdict
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


def _assign_columns(nodes: list, flows: list) -> dict[str, int]:
    """
    BFS-based column assignment.
    Nodes at the same depth from the start share the same column level,
    so parallel branches (e.g., two paths out of a DECISION) stack
    vertically instead of spreading out horizontally.
    """
    outgoing: dict[str, list[str]] = {n.get("id", ""): [] for n in nodes}
    for flow in flows:
        src = flow.get("sourceNodeId", "")
        tgt = flow.get("targetNodeId", "")
        if src in outgoing:
            outgoing[src].append(tgt)

    # Prefer explicit INITIAL_NODE; fallback to nodes with no incoming edge
    target_ids = {f.get("targetNodeId", "") for f in flows}
    start_ids = [n.get("id", "") for n in nodes if n.get("type") == "INITIAL_NODE"]
    if not start_ids:
        start_ids = [n.get("id", "") for n in nodes if n.get("id", "") not in target_ids]

    col: dict[str, int] = {}
    queue: deque = deque()
    for sid in start_ids:
        if sid not in col:
            col[sid] = 0
            queue.append(sid)

    while queue:
        nid = queue.popleft()
        for tgt in outgoing.get(nid, []):
            new_col = col[nid] + 1
            if tgt not in col or col[tgt] < new_col:
                col[tgt] = new_col
                queue.append(tgt)

    # Fallback: disconnected nodes get column 0
    for n in nodes:
        nid = n.get("id", "")
        if nid not in col:
            col[nid] = 0

    return col


def generate_bpmn_xml(data: dict) -> str:
    partitions = data.get("partitions", [])
    nodes = data.get("nodes", [])
    flows = data.get("flows", [])

    # Layout constants
    PARTICIPANT_X = 130
    PARTICIPANT_Y = 80
    LANE_LABEL_W = 30        # bpmn-js pool header width
    LANE_INNER_X = PARTICIPANT_X + LANE_LABEL_W  # x where lanes start = 160
    COL_W = 220              # horizontal spacing between nodes
    NODE_START_X = LANE_INNER_X + 70  # first node column, past lane label

    # ── BFS column assignment & slot grouping ─────────────────────────────────
    # Parallel branches share the same column so they stack vertically.
    node_cols = _assign_columns(nodes, flows)

    slot_map: dict[tuple, list] = defaultdict(list)
    for _node in nodes:
        _nid = _node.get("id", "")
        _pid = _node.get("partitionId", "")
        _part_idx = next(
            (j for j, p in enumerate(partitions) if p.get("id") == _pid), 0
        )
        slot_map[(_part_idx, node_cols.get(_nid, 0))].append(_node)

    max_stack = max((len(v) for v in slot_map.values()), default=1)
    max_col = max(node_cols.values(), default=0)

    # Dynamically size lane height to accommodate stacked nodes
    LANE_H = max(200, max_stack * 110)
    total_width = max(900, NODE_START_X - PARTICIPANT_X + (max_col + 1) * COL_W + 80)
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
            # name makes the condition text visible as a label on the arrow
            process_elements += (
                f'<bpmn:sequenceFlow id="{fid}" sourceRef="{src}" targetRef="{tgt}"'
                f' name="{_xe(cond)}">'
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

    # ── Pre-calculate node bounds (slot-based, supports vertical stacking) ──────
    node_bounds: dict[str, tuple[int, int, int, int]] = {}
    for (_part_idx, _col_idx), slot in slot_map.items():
        num = len(slot)
        for sub_idx, _node in enumerate(slot):
            _ntype = _node.get("type", "ACTION")
            _nid = _node.get("id", "")
            if _ntype in ("INITIAL_NODE", "ACTIVITY_FINAL"):
                _w, _h = 36, 36
            elif _ntype in ("DECISION", "MERGE", "FORK", "JOIN"):
                _w, _h = 50, 50
            else:
                _w, _h = 100, 80
            _nx = NODE_START_X + _col_idx * COL_W
            lane_y = PARTICIPANT_Y + _part_idx * LANE_H
            if num == 1:
                _ny = lane_y + (LANE_H - _h) // 2
            else:
                # Distribute nodes evenly within the lane height
                slot_h = LANE_H / num
                _ny = int(lane_y + sub_idx * slot_h + (slot_h - _h) / 2)
            node_bounds[_nid] = (_nx, _ny, _w, _h)

    # ── DI: node shapes ───────────────────────────────────────────────────────
    node_shapes = ""
    for node in nodes:
        nid = node.get("id", "")
        nx, ny, w, h = node_bounds[nid]
        node_shapes += (
            f'<bpmndi:BPMNShape id="{nid}_di" bpmnElement="{nid}">\n'
            f'  <dc:Bounds x="{nx}" y="{ny}" width="{w}" height="{h}"/>\n'
            f'</bpmndi:BPMNShape>\n'
        )

    # ── DI: edges with waypoints (prevents erratic routing in bpmn-js) ────────
    # Same-lane:   right-center → left-center  (straight)
    # Cross-lane:  right-center → mid-x,same-y → mid-x,target-y → left-center
    _node_map = {n.get("id"): n for n in nodes}
    edges = ""
    for flow in flows:
        fid = flow.get("id", "")
        src_id = flow.get("sourceNodeId", "")
        tgt_id = flow.get("targetNodeId", "")
        cond = flow.get("guardCondition")
        src_b = node_bounds.get(src_id)
        tgt_b = node_bounds.get(tgt_id)
        waypoints = ""
        label_xml = ""
        if src_b and tgt_b:
            sx, sy, sw, sh = src_b
            tx, ty, tw, th = tgt_b
            wp_sx, wp_sy = sx + sw, sy + sh // 2
            wp_tx, wp_ty = tx, ty + th // 2
            same_lane = (
                _node_map.get(src_id, {}).get("partitionId")
                == _node_map.get(tgt_id, {}).get("partitionId")
            )
            if same_lane:
                waypoints = (
                    f'<di:waypoint x="{wp_sx}" y="{wp_sy}"/>'
                    f'<di:waypoint x="{wp_tx}" y="{wp_ty}"/>'
                )
            else:
                mid_x = (wp_sx + wp_tx) // 2
                waypoints = (
                    f'<di:waypoint x="{wp_sx}" y="{wp_sy}"/>'
                    f'<di:waypoint x="{mid_x}" y="{wp_sy}"/>'
                    f'<di:waypoint x="{mid_x}" y="{wp_ty}"/>'
                    f'<di:waypoint x="{wp_tx}" y="{wp_ty}"/>'
                )
            if cond:
                lx = (wp_sx + wp_tx) // 2 - 50
                ly = (wp_sy + wp_ty) // 2 - 14
                label_xml = (
                    f'<bpmndi:BPMNLabel>'
                    f'<dc:Bounds x="{lx}" y="{ly}" width="100" height="14"/>'
                    f'</bpmndi:BPMNLabel>'
                )
        edges += (
            f'<bpmndi:BPMNEdge id="{fid}_di" bpmnElement="{fid}">'
            f'{waypoints}{label_xml}'
            f'</bpmndi:BPMNEdge>\n'
        )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" '
        'xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI" '
        'xmlns:dc="http://www.omg.org/spec/DD/20100524/DC" '
        'xmlns:di="http://www.omg.org/spec/DD/20100524/DI" '
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
