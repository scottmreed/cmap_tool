import streamlit as st
from typing import List, Tuple

from concept_map import ConceptMapBuilder


st.set_page_config(page_title="Concept Map Builder", layout="wide")
builder = ConceptMapBuilder()


def init_state() -> None:
    if "concepts" not in st.session_state:
        st.session_state.concepts: List[dict] = []
    if "edges" not in st.session_state:
        st.session_state.edges: List[Tuple[str, str]] = []
    if "next_id" not in st.session_state:
        st.session_state.next_id = 1


def reset_map() -> None:
    st.session_state.concepts = []
    st.session_state.edges = []
    st.session_state.next_id = 1


init_state()

st.title("Concept Map Builder")
st.caption(
    "Add concepts, attach them to existing items, preview the map, and export a polished PPTX."
)

with st.sidebar:
    st.header("Add Concept")
    with st.form("concept_form", clear_on_submit=True):
        concept_label = st.text_input("Concept text", placeholder="e.g. Systems Thinking")
        options = [concept["id"] for concept in st.session_state.concepts]
        option_labels = {concept["id"]: concept["label"] for concept in st.session_state.concepts}
        attached_to = st.multiselect(
            "Attach to existing concepts",
            options=options,
            format_func=lambda concept_id: option_labels.get(concept_id, concept_id),
            help="Select zero or more existing concepts to connect to this one.",
        )
        add_submitted = st.form_submit_button("Add Concept")
        if add_submitted:
            label = concept_label.strip()
            if not label:
                st.warning("Please enter text for the concept.")
            else:
                new_id = f"concept-{st.session_state.next_id}"
                st.session_state.next_id += 1
                st.session_state.concepts.append({"id": new_id, "label": label})
                for target in attached_to:
                    edge = (new_id, target)
                    if edge not in st.session_state.edges:
                        st.session_state.edges.append(edge)
                st.success(f"Added concept: {label}")

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

        link_submitted = st.form_submit_button("Create Link", disabled=not has_pairs)
        if link_submitted and source and target:
            if source == target:
                st.warning("Cannot link a concept to itself.")
            else:
                edge = (source, target)
                if edge in st.session_state.edges:
                    st.info("That link already exists.")
                else:
                    st.session_state.edges.append(edge)
                    st.success("Link added.")

    if st.button("Reset Map", type="primary", use_container_width=True):
        reset_map()
        st.experimental_rerun()

if st.session_state.concepts:
    layout = {}
    col_left, col_right = st.columns([2, 1])
    with col_left:
        try:
            layout = builder.compute_layout(st.session_state.concepts, st.session_state.edges)
            preview_img = builder.render_preview(layout, st.session_state.edges)
            st.image(preview_img, caption="Current layout preview", use_container_width=True)
        except Exception as err:
            st.error(f"Unable to render concept map: {err}")
            layout = {}

    with col_right:
        st.subheader("Concepts")
        st.dataframe(
            [{"Label": concept["label"], "ID": concept["id"]} for concept in st.session_state.concepts],
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Connections")
        if st.session_state.edges:
            connection_rows = []
            label_lookup = {concept["id"]: concept["label"] for concept in st.session_state.concepts}
            for src, dst in st.session_state.edges:
                connection_rows.append(
                    {"From": label_lookup.get(src, src), "To": label_lookup.get(dst, dst)}
                )
            st.dataframe(connection_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No connections yet.")

    if st.session_state.concepts and st.session_state.edges and layout:
        pptx_bytes = builder.build_pptx_bytes(st.session_state.concepts, st.session_state.edges, layout=layout)
        st.download_button(
            "Download PPTX",
            data=pptx_bytes,
            file_name="concept_map.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    elif st.session_state.concepts:
        st.info("Add at least one connection to enable PPTX export.")
else:
    st.info("Start by adding your first concept using the sidebar controls.")
