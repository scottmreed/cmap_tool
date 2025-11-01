import streamlit as st
from typing import List, Dict
import json
import streamlit.components.v1 as components

from concept_map import ConceptMapBuilder


st.set_page_config(page_title="Concept Map Builder", layout="wide")
builder = ConceptMapBuilder()


def init_state() -> None:
    if "concepts" not in st.session_state:
        st.session_state.concepts: List[dict] = []
    if "edges" not in st.session_state:
        st.session_state.edges: List[Dict[str, str]] = []
    if "next_id" not in st.session_state:
        st.session_state.next_id = 1
    if "map_type" not in st.session_state:
        st.session_state.map_type = "concept_map"
    if "previous_map_type" not in st.session_state:
        st.session_state.previous_map_type = "concept_map"
    if "pending_connection" not in st.session_state:
        st.session_state.pending_connection = None  # Dict with "source" and "target" keys


def reset_map() -> None:
    st.session_state.concepts = []
    st.session_state.edges = []
    st.session_state.next_id = 1
    st.session_state.pending_connection = None


def create_interactive_map(layout: Dict, edges: List[Dict[str, str]], concepts: List[dict], is_concept_map: bool) -> str:
    """Create interactive HTML/JavaScript component for drag-and-drop connections."""
    # Convert layout to JSON-serializable format
    layout_data = {}
    for node_id, node in layout.items():
        layout_data[node_id] = {
            "center": node.center_in,
            "size": (node.width_in, node.height_in),
            "label": "\n".join(node.lines)  # Join lines for display
        }
    
    # Convert edges to simpler format
    edges_data = []
    for edge in edges:
        if isinstance(edge, dict):
            edges_data.append({
                "source": edge.get("source"),
                "target": edge.get("target"),
                "label": edge.get("label", "")
            })
    
    # Convert concepts for lookup
    concepts_data = {c["id"]: c["label"] for c in concepts}
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }}
            #canvas-container {{
                position: relative;
                width: 100%;
                height: 100%;
                min-height: 500px;
                background: white;
                border: 1px solid #ddd;
            }}
            canvas {{
                display: block;
                width: 100%;
                height: 100%;
                cursor: default;
            }}
            .concept-node {{
                cursor: grab;
                user-select: none;
            }}
            .concept-node:active {{
                cursor: grabbing;
            }}
            .concept-node.dragging {{
                cursor: grabbing;
            }}
        </style>
    </head>
    <body>
        <div id="canvas-container">
            <canvas id="mapCanvas"></canvas>
        </div>
        <script>
            const layoutData = {json.dumps(layout_data)};
            const edgesData = {json.dumps(edges_data)};
            const conceptsData = {json.dumps(concepts_data)};
            const isConceptMap = {str(is_concept_map).lower()};
            const isInteractive = isConceptMap;
            
            const canvas = document.getElementById('mapCanvas');
            const container = document.getElementById('canvas-container');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size
            function resizeCanvas() {{
                const rect = container.getBoundingClientRect();
                canvas.width = rect.width;
                canvas.height = Math.max(500, rect.height);
            }}
            
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);
            
            // Scale to fit the slide dimensions (13.333 x 7.5 inches)
            const slideWidth = 13.333;
            const slideHeight = 7.5;
            const padding = 0.5;
            
            function getScale() {{
                const scaleX = (canvas.width - padding * 2) / slideWidth;
                const scaleY = (canvas.height - padding * 2) / slideHeight;
                return Math.min(scaleX, scaleY);
            }}
            
            function toCanvasX(x) {{
                const scale = getScale();
                return padding + x * scale;
            }}
            
            function toCanvasY(y) {{
                const scale = getScale();
                return padding + y * scale;
            }}
            
            // Draw functions
            function drawArrow(ctx, x1, y1, x2, y2, label) {{
                ctx.save();
                ctx.strokeStyle = '#7a7a7a';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(toCanvasX(x1), toCanvasY(y1));
                ctx.lineTo(toCanvasX(x2), toCanvasY(y2));
                ctx.stroke();
                
                // Arrowhead
                const angle = Math.atan2(y2 - y1, x2 - x1);
                const arrowLength = 8 / getScale();
                const arrowAngle = Math.PI / 6;
                ctx.beginPath();
                ctx.moveTo(toCanvasX(x2), toCanvasY(y2));
                ctx.lineTo(
                    toCanvasX(x2 - arrowLength * Math.cos(angle - arrowAngle)),
                    toCanvasY(y2 - arrowLength * Math.sin(angle - arrowAngle))
                );
                ctx.moveTo(toCanvasX(x2), toCanvasY(y2));
                ctx.lineTo(
                    toCanvasX(x2 - arrowLength * Math.cos(angle + arrowAngle)),
                    toCanvasY(y2 - arrowLength * Math.sin(angle + arrowAngle))
                );
                ctx.stroke();
                
                // Label
                if (label && label.trim()) {{
                    const midX = (x1 + x2) / 2;
                    const midY = (y1 + y2) / 2;
                    ctx.fillStyle = 'white';
                    ctx.strokeStyle = '#787878';
                    ctx.lineWidth = 1;
                    const textWidth = ctx.measureText(label.trim()).width;
                    const textHeight = 12;
                    const padding = 4;
                    ctx.fillRect(
                        toCanvasX(midX) - textWidth/2 - padding,
                        toCanvasY(midY) - textHeight/2 - padding,
                        textWidth + padding * 2,
                        textHeight + padding * 2
                    );
                    ctx.strokeRect(
                        toCanvasX(midX) - textWidth/2 - padding,
                        toCanvasY(midY) - textHeight/2 - padding,
                        textWidth + padding * 2,
                        textHeight + padding * 2
                    );
                    ctx.fillStyle = '#323232';
                    ctx.font = '7px Calibri';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(label.trim(), toCanvasX(midX), toCanvasY(midY));
                }}
                ctx.restore();
            }}
            
            function drawNode(ctx, nodeId, x, y, width, height, label, isDragging, isHovered) {{
                ctx.save();
                
                // Node background
                ctx.fillStyle = isDragging ? '#e0e0e0' : (isHovered && isInteractive ? '#f0f0f0' : '#f5f5f5');
                ctx.strokeStyle = '#787878';
                ctx.lineWidth = isDragging ? 2 : 1.5;
                
                const rx = toCanvasX(x - width/2);
                const ry = toCanvasY(y - height/2);
                const rw = width * getScale();
                const rh = height * getScale();
                const radius = 5;
                
                ctx.beginPath();
                ctx.moveTo(rx + radius, ry);
                ctx.lineTo(rx + rw - radius, ry);
                ctx.quadraticCurveTo(rx + rw, ry, rx + rw, ry + radius);
                ctx.lineTo(rx + rw, ry + rh - radius);
                ctx.quadraticCurveTo(rx + rw, ry + rh, rx + rw - radius, ry + rh);
                ctx.lineTo(rx + radius, ry + rh);
                ctx.quadraticCurveTo(rx, ry + rh, rx, ry + rh - radius);
                ctx.lineTo(rx, ry + radius);
                ctx.quadraticCurveTo(rx, ry, rx + radius, ry);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
                
                // Text
                ctx.fillStyle = '#323232';
                ctx.font = '9px Calibri';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                // Split label by newlines (JSON.stringify handles escaping, so this should work)
                const lines = label.split('\n');
                const lineHeight = 11;
                const startY = ry + rh/2 - (lines.length - 1) * lineHeight / 2;
                lines.forEach((line, i) => {{
                    ctx.fillText(line, rx + rw/2, startY + i * lineHeight);
                }});
                
                ctx.restore();
            }}
            
            function redraw() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Draw edges
                edgesData.forEach(edge => {{
                    if (layoutData[edge.source] && layoutData[edge.target]) {{
                        const src = layoutData[edge.source];
                        const dst = layoutData[edge.target];
                        drawArrow(ctx, src.center[0], src.center[1], dst.center[0], dst.center[1], edge.label);
                    }}
                }});
                
                // Draw temporary connection line if dragging
                if (dragState.dragging && dragState.sourceId) {{
                    const src = layoutData[dragState.sourceId];
                    if (src) {{
                        ctx.save();
                        ctx.strokeStyle = '#4CAF50';
                        ctx.lineWidth = 2;
                        ctx.setLineDash([5, 5]);
                        ctx.beginPath();
                        ctx.moveTo(toCanvasX(src.center[0]), toCanvasY(src.center[1]));
                        ctx.lineTo(dragState.currentX, dragState.currentY);
                        ctx.stroke();
                        ctx.restore();
                    }}
                }}
                
                // Draw nodes
                Object.keys(layoutData).forEach(nodeId => {{
                    const node = layoutData[nodeId];
                    const isDragging = dragState.dragging && dragState.sourceId === nodeId;
                    const isHovered = hoveredNode === nodeId;
                    drawNode(
                        ctx, nodeId,
                        node.center[0], node.center[1],
                        node.size[0], node.size[1],
                        node.label,
                        isDragging, isHovered
                    );
                }});
            }}
            
            // Drag and drop state
            const dragState = {{
                dragging: false,
                sourceId: null,
                startX: 0,
                startY: 0,
                currentX: 0,
                currentY: 0
            }};
            
            let hoveredNode = null;
            
            function getNodeAt(x, y) {{
                const scale = getScale();
                const slideX = (x - padding) / scale;
                const slideY = (y - padding) / scale;
                
                for (const [nodeId, node] of Object.entries(layoutData)) {{
                    const nodeX = node.center[0];
                    const nodeY = node.center[1];
                    const nodeW = node.size[0];
                    const nodeH = node.size[1];
                    
                    if (slideX >= nodeX - nodeW/2 && slideX <= nodeX + nodeW/2 &&
                        slideY >= nodeY - nodeH/2 && slideY <= nodeY + nodeH/2) {{
                        return nodeId;
                    }}
                }}
                return null;
            }}
            
            function getCanvasCoordinates(event) {{
                const rect = canvas.getBoundingClientRect();
                return {{
                    x: event.clientX - rect.left,
                    y: event.clientY - rect.top
                }};
            }}
            
            canvas.addEventListener('mousedown', (e) => {{
                if (!isInteractive) return;
                
                const coords = getCanvasCoordinates(e);
                const nodeId = getNodeAt(coords.x, coords.y);
                
                if (nodeId) {{
                    dragState.dragging = true;
                    dragState.sourceId = nodeId;
                    dragState.startX = coords.x;
                    dragState.startY = coords.y;
                    dragState.currentX = coords.x;
                    dragState.currentY = coords.y;
                    canvas.style.cursor = 'grabbing';
                    redraw();
                }}
            }});
            
            canvas.addEventListener('mousemove', (e) => {{
                const coords = getCanvasCoordinates(e);
                const nodeId = getNodeAt(coords.x, coords.y);
                
                if (dragState.dragging) {{
                    dragState.currentX = coords.x;
                    dragState.currentY = coords.y;
                    canvas.style.cursor = 'grabbing';
                    redraw();
                }} else if (isInteractive) {{
                    if (nodeId) {{
                        canvas.style.cursor = 'grab';
                        hoveredNode = nodeId;
                    }} else {{
                        canvas.style.cursor = 'default';
                        hoveredNode = null;
                    }}
                    redraw();
                }}
            }});
            
            canvas.addEventListener('mouseup', (e) => {{
                if (!dragState.dragging || !isInteractive) return;
                
                const coords = getCanvasCoordinates(e);
                const targetId = getNodeAt(coords.x, coords.y);
                
                if (targetId && targetId !== dragState.sourceId) {{
                    // Check if connection already exists
                    const exists = edgesData.some(edge => 
                        edge.source === dragState.sourceId && edge.target === targetId
                    );
                    
                    if (!exists) {{
                        // Communicate with Streamlit parent frame
                        try {{
                            // Try to access Streamlit's communication mechanism
                            if (window.parent && window.parent.streamlit) {{
                                window.parent.streamlit.setComponentValue({{
                                    source: dragState.sourceId,
                                    target: targetId
                                }});
                            }} else {{
                                // Fallback: use postMessage to parent
                                window.parent.postMessage({{
                                    type: 'concept_map_connection',
                                    source: dragState.sourceId,
                                    target: targetId
                                }}, '*');
                            }}
                        }} catch (e) {{
                            console.error('Error communicating with Streamlit:', e);
                            // Fallback: try to set URL parameters on parent
                            try {{
                                const params = new URLSearchParams(window.top.location.search);
                                params.set('create_connection', 'true');
                                params.set('source', dragState.sourceId);
                                params.set('target', targetId);
                                window.top.location.search = params.toString();
                            }} catch (e2) {{
                                console.error('Fallback also failed:', e2);
                            }}
                        }}
                    }}
                }}
                
                dragState.dragging = false;
                dragState.sourceId = null;
                canvas.style.cursor = 'default';
                redraw();
            }});
            
            canvas.addEventListener('mouseleave', () => {{
                if (dragState.dragging) {{
                    dragState.dragging = false;
                    dragState.sourceId = null;
                    canvas.style.cursor = 'default';
                    redraw();
                }}
                hoveredNode = null;
            }});
            
            // Initial draw
            redraw();
        </script>
    </body>
    </html>
    """
    return html


def get_edge_key(edge: Dict[str, str]) -> tuple:
    """Get a unique key for an edge (source, target) for comparison."""
    return (edge["source"], edge["target"])


def edge_exists(source: str, target: str, edges: List[Dict[str, str]]) -> bool:
    """Check if an edge from source to target already exists."""
    return any(e["source"] == source and e["target"] == target for e in edges)


def delete_concept(concept_id: str) -> None:
    """Delete a concept and all associated edges."""
    # Remove the concept
    st.session_state.concepts = [c for c in st.session_state.concepts if c["id"] != concept_id]
    # Remove all edges connected to this concept
    st.session_state.edges = [
        e for e in st.session_state.edges
        if e["source"] != concept_id and e["target"] != concept_id
    ]


def delete_edge(source: str, target: str) -> None:
    """Delete an edge between two concepts."""
    st.session_state.edges = [
        e for e in st.session_state.edges
        if not (e["source"] == source and e["target"] == target)
    ]


def convert_concept_map_to_mind_map() -> None:
    """Convert connection labels to connector concepts when switching to mind map."""
    # Build list of edges that need conversion
    edges_with_labels = []
    for i, edge in enumerate(st.session_state.edges):
        edge_label = edge.get("label", "").strip()
        if edge_label:
            edges_with_labels.append({
                "index": i,
                "edge": edge,
                "label": edge_label,
                "source": edge["source"],
                "target": edge["target"]
            })
    
    if not edges_with_labels:
        return
    
    # Create connector concepts and new edges
    new_edges = []
    connector_mapping = {}  # Maps (source, target, label) -> connector_id
    
    for item in edges_with_labels:
        # Create a connector concept for this edge label
        connector_id = f"connector-{st.session_state.next_id}"
        st.session_state.next_id += 1
        connector_label = f"connector: {item['label']}"
        st.session_state.concepts.append({"id": connector_id, "label": connector_label})
        
        # Map this edge to its connector
        edge_key = (item["source"], item["target"], item["label"])
        connector_mapping[edge_key] = connector_id
        
        # Create edge from source to connector (no label)
        new_edges.append({
            "source": item["source"],
            "target": connector_id,
            "label": ""
        })
        # Create edge from connector to target (no label)
        new_edges.append({
            "source": connector_id,
            "target": item["target"],
            "label": ""
        })
    
    # Keep edges without labels as-is
    for edge in st.session_state.edges:
        edge_label = edge.get("label", "").strip()
        if not edge_label:
            new_edges.append(edge)
    
    st.session_state.edges = new_edges


def convert_mind_map_to_concept_map() -> None:
    """Convert connector concepts back to connection labels when switching to concept map."""
    connector_concepts = []
    connector_ids = set()
    
    # Find all connector concepts
    for concept in st.session_state.concepts:
        label = concept.get("label", "")
        if label.startswith("connector: "):
            connector_concepts.append(concept)
            connector_ids.add(concept["id"])
    
    if not connector_concepts:
        return
    
    # Build a map of connector -> (incoming_edges, outgoing_edges)
    connector_edges = {}
    for connector_id in connector_ids:
        connector_edges[connector_id] = {
            "incoming": [],
            "outgoing": []
        }
    
    # Separate edges: those involving connectors and those that don't
    regular_edges = []
    for edge in st.session_state.edges:
        source = edge.get("source")
        target = edge.get("target")
        
        if source in connector_ids:
            # Edge goes from connector to target
            connector_edges[source]["outgoing"].append(edge)
        elif target in connector_ids:
            # Edge goes from source to connector
            connector_edges[target]["incoming"].append(edge)
        else:
            # Regular edge, keep it
            regular_edges.append(edge)
    
    # For each connector, pair up incoming and outgoing edges
    edges_to_add = []
    for connector_id, edges_info in connector_edges.items():
        incoming = edges_info["incoming"]
        outgoing = edges_info["outgoing"]
        
        # Find the connector concept to get the label
        connector_concept = next((c for c in connector_concepts if c["id"] == connector_id), None)
        if connector_concept:
            original_label = connector_concept["label"].replace("connector: ", "", 1).strip()
            
            # Create edges for each incoming-outgoing pair
            for in_edge in incoming:
                for out_edge in outgoing:
                    edges_to_add.append({
                        "source": in_edge["source"],
                        "target": out_edge["target"],
                        "label": original_label
                    })
    
    # Update edges: remove connector-related edges and add labeled edges
    st.session_state.edges = regular_edges
    
    # Add edges with labels
    for edge_to_add in edges_to_add:
        # Check if edge already exists
        if not edge_exists(edge_to_add["source"], edge_to_add["target"], st.session_state.edges):
            st.session_state.edges.append(edge_to_add)
    
    # Remove connector concepts
    st.session_state.concepts = [
        c for c in st.session_state.concepts 
        if c["id"] not in connector_ids
    ]


def update_edge(old_source: str, old_target: str, new_source: str, new_target: str, new_label: str) -> None:
    """Update an edge's source, target, and label."""
    for edge in st.session_state.edges:
        if edge["source"] == old_source and edge["target"] == old_target:
            edge["source"] = new_source
            edge["target"] = new_target
            edge["label"] = new_label
            break


def validate_concept_map(concepts: List[dict], edges: List[Dict[str, str]]) -> tuple[bool, List[str]]:
    """Validate that concept map requirements are met. Returns (is_valid, errors)."""
    errors = []
    
    if not concepts:
        return True, []  # Empty map is valid
    
    concept_ids = {c["id"] for c in concepts}
    
    # Check if all concepts are connected
    connected_ids = set()
    for edge in edges:
        connected_ids.add(edge["source"])
        connected_ids.add(edge["target"])
    
    unconnected = concept_ids - connected_ids
    if unconnected:
        unconnected_labels = [c["label"] for c in concepts if c["id"] in unconnected]
        errors.append(f"All concepts must be connected. Unconnected concepts: {', '.join(unconnected_labels)}")
    
    # Check if all edges have labels
    unlabeled_edges = []
    for edge in edges:
        if not edge.get("label") or not edge["label"].strip():
            source_label = next((c["label"] for c in concepts if c["id"] == edge["source"]), edge["source"])
            target_label = next((c["label"] for c in concepts if c["id"] == edge["target"]), edge["target"])
            unlabeled_edges.append(f"{source_label} → {target_label}")
    
    if unlabeled_edges:
        errors.append(f"All connections must have a label/term. Missing labels for: {', '.join(unlabeled_edges[:5])}")
        if len(unlabeled_edges) > 5:
            errors[-1] += f" (and {len(unlabeled_edges) - 5} more)"
    
    return len(errors) == 0, errors


init_state()

# Handle connection creation from drag-and-drop
try:
    if hasattr(st, 'query_params'):
        query_params = st.query_params
        create_conn = query_params.get("create_connection")
        if create_conn:
            source = query_params.get("source")
            target = query_params.get("target")
            # Handle both list and single value returns
            if isinstance(source, list):
                source = source[0] if source else None
            if isinstance(target, list):
                target = target[0] if target else None
            if source and target and not edge_exists(source, target, st.session_state.edges):
                st.session_state.pending_connection = {"source": source, "target": target}
                # Clear query params by setting them to empty
                query_params["create_connection"] = None
                query_params["source"] = None
                query_params["target"] = None
                st.rerun()
except Exception:
    pass  # query_params might not be available in older Streamlit versions

# Handle pending connection form submission
if st.session_state.pending_connection:
    source = st.session_state.pending_connection["source"]
    target = st.session_state.pending_connection["target"]
    source_label = next((c["label"] for c in st.session_state.concepts if c["id"] == source), source)
    target_label = next((c["label"] for c in st.session_state.concepts if c["id"] == target), target)
    
    with st.form("pending_connection_form", clear_on_submit=True):
        st.info(f"Creating connection: **{source_label}** → **{target_label}**")
        connection_label = st.text_input(
            "Connection label/term",
            placeholder="e.g., contains, leads to, requires",
            help="Required for concept maps, optional for mind maps",
            key="pending_connection_label"
        )
        col_create, col_cancel = st.columns(2)
        with col_create:
            create_submitted = st.form_submit_button("Create Connection", type="primary", use_container_width=True)
        with col_cancel:
            cancel_submitted = st.form_submit_button("Cancel", use_container_width=True)
        
        if create_submitted:
            if is_concept_map and not connection_label.strip():
                st.warning("Concept maps require a label for each connection.")
            else:
                st.session_state.edges.append({
                    "source": source,
                    "target": target,
                    "label": connection_label.strip()
                })
                st.session_state.pending_connection = None
                st.success("Connection created!")
                st.rerun()
        
        if cancel_submitted:
            st.session_state.pending_connection = None
            st.rerun()

# Add JavaScript listener for drag-and-drop connections from iframe
st.markdown("""
<script>
// Listen for connection messages from iframe
window.addEventListener('message', function(event) {
    if (event.data && event.data.type === 'concept_map_connection') {
        const source = event.data.source;
        const target = event.data.target;
        if (source && target) {
            // Set query parameters to trigger Streamlit rerun
            const url = new URL(window.location);
            url.searchParams.set('create_connection', 'true');
            url.searchParams.set('source', source);
            url.searchParams.set('target', target);
            window.location.href = url.toString();
        }
    }
});
</script>
""", unsafe_allow_html=True)

# Remove "press enter to submit form" text if it appears
st.markdown("""
<script>
(function() {
    function removeEnterSubmitText() {
        // Find and remove any text containing "press enter to submit" (case insensitive)
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        const textNodes = [];
        let node;
        while (node = walker.nextNode()) {
            const text = node.textContent.toLowerCase();
            if (text.includes('press enter') && text.includes('submit')) {
                textNodes.push(node);
            }
        }
        
        textNodes.forEach(textNode => {
            // Remove the entire text node if it matches
            if (textNode.textContent.toLowerCase().includes('press enter') && 
                textNode.textContent.toLowerCase().includes('submit')) {
                textNode.remove();
            } else {
                // Otherwise, try to clean up the text
                const parent = textNode.parentElement;
                if (parent) {
                    textNode.textContent = textNode.textContent
                        .replace(/press enter to submit form/gi, '')
                        .replace(/press enter to submit/gi, '');
                    if (!textNode.textContent.trim()) {
                        textNode.remove();
                    }
                }
            }
        });
        
        // Also check for help text or small text elements
        const allElements = document.querySelectorAll('*');
        allElements.forEach(el => {
            const text = el.textContent || el.innerText || '';
            if (text.toLowerCase().includes('press enter') && 
                text.toLowerCase().includes('submit')) {
                // Try to remove just that text
                if (el.textContent) {
                    el.textContent = el.textContent
                        .replace(/press enter to submit form/gi, '')
                        .replace(/press enter to submit/gi, '');
                }
                if (el.innerText) {
                    el.innerText = el.innerText
                        .replace(/press enter to submit form/gi, '')
                        .replace(/press enter to submit/gi, '');
                }
            }
        });
    }
    
    // Run on page load and after updates
    function runCleanup() {
        removeEnterSubmitText();
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', runCleanup);
    } else {
        runCleanup();
    }
    
    // Also run after Streamlit updates
    const observer = new MutationObserver(runCleanup);
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
})();
</script>
""", unsafe_allow_html=True)

st.title("Concept Map Builder")

# Map type toggle
map_type = st.radio(
    "Map Type",
    ["Concept Map", "Mind Map"],
    index=0 if st.session_state.map_type == "concept_map" else 1,
    horizontal=True,
    help="Concept maps require all concepts to be connected and all connections to have labels. Mind maps allow disconnected concepts and unlabeled connections."
)
new_map_type = "concept_map" if map_type == "Concept Map" else "mind_map"

# Handle conversion when map type changes
if new_map_type != st.session_state.map_type:
    if new_map_type == "mind_map":
        # Converting from concept map to mind map
        convert_concept_map_to_mind_map()
    else:
        # Converting from mind map to concept map
        convert_mind_map_to_concept_map()
    st.session_state.previous_map_type = st.session_state.map_type
    st.session_state.map_type = new_map_type
    st.rerun()

is_concept_map = st.session_state.map_type == "concept_map"

st.caption(
    "Add concepts, attach them to existing items, preview the map, and export a polished PPTX. "
    + ("**Tip:** Click and drag from one concept to another to create a connection!" if is_concept_map else "")
)

with st.sidebar:
    st.header("Add Concept")
    with st.form("concept_form", clear_on_submit=True):
        concept_label = st.text_input("Concept text", placeholder="e.g. Systems Thinking")
        # Always list all available concepts for linking (even if empty)
        options = [concept["id"] for concept in st.session_state.concepts]
        option_labels = {concept["id"]: concept["label"] for concept in st.session_state.concepts}
        
        # Always show multiselect for attaching to existing concepts (disabled when no concepts exist)
        attached_to = st.multiselect(
            "Attach to existing concepts",
            options=options,
            format_func=lambda concept_id: option_labels.get(concept_id, concept_id),
            help="Select zero or more existing concepts to connect to this one. An arrow will be created from the new concept to each selected concept.",
            disabled=len(options) == 0,
        )
        
        # Always show connection label field when there are existing concepts (available from first concept onwards)
        connection_label = ""
        if len(options) >= 1:
            connection_label = st.text_input(
                "Connection label/term",
                placeholder="e.g., contains, leads to, requires",
                help="Label for the connection(s). This label will be applied to all connections created. Required for concept maps, optional for mind maps.",
                key="add_concept_connection_label"
            )
        add_submitted = st.form_submit_button("Add Concept")
        if add_submitted:
            label = concept_label.strip()
            if not label:
                st.warning("Please enter text for the concept.")
            else:
                # Check for duplicate concept label
                existing_labels = [c["label"].lower() for c in st.session_state.concepts]
                if label.lower() in existing_labels:
                    st.warning(f"⚠️ A concept with the label '{label}' already exists. Adding duplicate anyway.")
                
                new_id = f"concept-{st.session_state.next_id}"
                st.session_state.next_id += 1
                st.session_state.concepts.append({"id": new_id, "label": label})
                # Create arrows (edges) when concepts are selected for attachment
                if attached_to:
                    # Validate connection label for concept maps
                    if is_concept_map and not connection_label.strip():
                        st.warning("⚠️ Concept maps require labels for all connections. Please enter a connection label/term.")
                    else:
                        edges_created = 0
                        for target in attached_to:
                            if not edge_exists(new_id, target, st.session_state.edges):
                                # Create edge (arrow) from new concept to existing concept
                                st.session_state.edges.append({
                                    "source": new_id,
                                    "target": target,
                                    "label": connection_label.strip() if connection_label else ""
                                })
                                edges_created += 1
                        if edges_created > 0:
                            if connection_label.strip():
                                st.success(f"✓ Added concept '{label}' and created {edges_created} arrow(s) with label '{connection_label.strip()}'.")
                            else:
                                st.success(f"✓ Added concept '{label}' and created {edges_created} arrow(s) to connect to existing concepts.")
                        else:
                            st.success(f"Added concept: {label}")
                else:
                    # No concepts selected for attachment - just add the concept
                    st.success(f"Added concept: {label}")
                    if len(st.session_state.concepts) > 1:
                        st.info("💡 Tip: Use 'Attach to existing concepts' to create arrows when adding a new concept.")

    st.header("Link Concepts")
    with st.form("link_form"):
        options = [concept["id"] for concept in st.session_state.concepts]
        option_labels = {concept["id"]: concept["label"] for concept in st.session_state.concepts}
        has_pairs = len(options) >= 2
        if has_pairs:
            source = st.selectbox(
                "From concept",
                options=options,
                format_func=lambda concept_id: option_labels.get(concept_id, concept_id),
            )
            target = st.selectbox(
                "To concept",
                options=options,
                format_func=lambda concept_id: option_labels.get(concept_id, concept_id),
                index=1 if len(options) > 1 else 0,
            )
        else:
            st.info("Add at least two concepts to create links.")
            source = target = None
        
        # Always show connection label field when there are concepts (available from first concept onwards)
        connection_label = ""
        if len(options) >= 1:
            connection_label = st.text_input(
                "Connection label/term",
                placeholder="e.g., contains, leads to, requires",
                help="Required for concept maps, optional for mind maps",
                disabled=not has_pairs,
            )

        link_submitted = st.form_submit_button("Create Link", disabled=not has_pairs)
        if link_submitted and source and target:
            if source == target:
                st.warning("Cannot link a concept to itself.")
            elif edge_exists(source, target, st.session_state.edges):
                st.info("That link already exists.")
            else:
                if is_concept_map and not connection_label.strip():
                    st.warning("Concept maps require a label for each connection.")
                else:
                    st.session_state.edges.append({
                        "source": source,
                        "target": target,
                        "label": connection_label.strip()
                    })
                    st.success("Link added.")

    if st.button("Reset Map", type="primary", use_container_width=True):
        reset_map()
        st.rerun()

# Validation for concept maps
validation_errors = []
if is_concept_map and st.session_state.concepts:
    is_valid, validation_errors = validate_concept_map(st.session_state.concepts, st.session_state.edges)
    if validation_errors:
        for error in validation_errors:
            st.error(error)

if st.session_state.concepts:
    layout = {}
    col_left, col_right = st.columns([2, 1])
    with col_left:
        try:
            layout = builder.compute_layout(st.session_state.concepts, st.session_state.edges)
            
            # Use interactive component for concept maps, static image for mind maps
            if is_concept_map:
                interactive_html = create_interactive_map(layout, st.session_state.edges, st.session_state.concepts, is_concept_map)
                components.html(interactive_html, height=600)
            else:
                # For mind maps, render without edge labels
                preview_img = builder.render_preview(layout, st.session_state.edges, show_edge_labels=False)
                st.image(preview_img, caption="Current layout preview", use_container_width=True)
        except Exception as err:
            st.error(f"Unable to render map: {err}")
            layout = {}

    with col_right:
        st.subheader("Concepts")
        if st.session_state.concepts:
            for i, concept in enumerate(st.session_state.concepts):
                col_label, col_delete = st.columns([4, 1])
                with col_label:
                    st.write(f"**{concept['label']}**")
                with col_delete:
                    if st.button("🗑️", key=f"delete_concept_{concept['id']}", help="Delete concept"):
                        delete_concept(concept["id"])
                        st.rerun()
        else:
            st.caption("No concepts yet.")
        
        st.subheader("Connections")
        if st.session_state.edges:
            label_lookup = {concept["id"]: concept["label"] for concept in st.session_state.concepts}
            for i, edge in enumerate(st.session_state.edges):
                source_label = label_lookup.get(edge["source"], edge["source"])
                target_label = label_lookup.get(edge["target"], edge["target"])
                edge_label = edge.get("label", "").strip() or "(no label)"
                
                with st.expander(f"{source_label} → {target_label}: {edge_label}", expanded=False):
                    # Edit connection form
                    with st.form(f"edit_edge_{i}", clear_on_submit=False):
                        options = [c["id"] for c in st.session_state.concepts]
                        option_labels = {c["id"]: c["label"] for c in st.session_state.concepts}
                        
                        new_source = st.selectbox(
                            "From concept",
                            options=options,
                            index=options.index(edge["source"]) if edge["source"] in options else 0,
                            format_func=lambda concept_id: option_labels.get(concept_id, concept_id),
                            key=f"edit_source_{i}"
                        )
                        new_target = st.selectbox(
                            "To concept",
                            options=options,
                            index=options.index(edge["target"]) if edge["target"] in options else 0,
                            format_func=lambda concept_id: option_labels.get(concept_id, concept_id),
                            key=f"edit_target_{i}"
                        )
                        new_label = st.text_input(
                            "Connection label/term",
                            value=edge.get("label", ""),
                            key=f"edit_label_{i}"
                        )
                        
                        col_update, col_delete = st.columns(2)
                        with col_update:
                            update_submitted = st.form_submit_button("Update", use_container_width=True)
                        with col_delete:
                            delete_submitted = st.form_submit_button("Delete", type="secondary", use_container_width=True)
                        
                        if update_submitted:
                            if new_source == new_target:
                                st.warning("Cannot link a concept to itself.")
                            else:
                                # Check if endpoints changed
                                endpoints_changed = (new_source != edge["source"] or new_target != edge["target"])
                                
                                # If endpoints changed, check if new connection already exists
                                if endpoints_changed:
                                    # Check if a different edge with these endpoints exists
                                    other_edges = [e for e in st.session_state.edges 
                                                  if not (e["source"] == edge["source"] and e["target"] == edge["target"])]
                                    if edge_exists(new_source, new_target, other_edges):
                                        st.warning("A connection with these endpoints already exists.")
                                    else:
                                        # Update edge (endpoints and label)
                                        update_edge(edge["source"], edge["target"], new_source, new_target, new_label.strip())
                                        st.success("Connection updated.")
                                        st.rerun()
                                else:
                                    # Only updating label, no endpoint change
                                    update_edge(edge["source"], edge["target"], edge["source"], edge["target"], new_label.strip())
                                    st.success("Connection updated.")
                                    st.rerun()
                        
                        if delete_submitted:
                            delete_edge(edge["source"], edge["target"])
                            st.success("Connection deleted.")
                            st.rerun()
        else:
            st.caption("No connections yet.")

    if st.session_state.concepts and st.session_state.edges and layout:
        # For concept maps, only allow export if validation passes
        if is_concept_map and validation_errors:
            st.info("Fix validation errors above before exporting.")
        else:
            file_prefix = "concept_map" if is_concept_map else "mind_map"
            
            # Export buttons in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pptx_bytes = builder.build_pptx_bytes(st.session_state.concepts, st.session_state.edges, layout=layout)
                st.download_button(
                    "Download PPTX",
                    data=pptx_bytes,
                    file_name=f"{file_prefix}.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                )
            
            with col2:
                jpg_bytes = builder.render_jpg(layout, st.session_state.edges)
                st.download_button(
                    "Download JPG",
                    data=jpg_bytes,
                    file_name=f"{file_prefix}.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
            
            with col3:
                cxl_bytes = builder.build_cxl_xml(
                    st.session_state.concepts,
                    st.session_state.edges,
                    layout=layout,
                    title=file_prefix.replace("_", " ").title(),
                )
                st.download_button(
                    "Download CXL",
                    data=cxl_bytes,
                    file_name=f"{file_prefix}.cxl",
                    mime="application/xml",
                    use_container_width=True,
                )
    elif st.session_state.concepts:
        st.info("Add at least one connection to enable export.")
else:
    st.info("Start by adding your first concept using the sidebar controls.")
