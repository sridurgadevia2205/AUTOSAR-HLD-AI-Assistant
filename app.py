import streamlit as st
import backend


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AUTOSAR HLD AI Assistant",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚗 AUTOSAR HLD AI Assistant")

st.write(
    "AI-powered analysis and question answering for "
    "AUTOSAR High-Level Design documents."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")


# ============================================================
# PDF UPLOAD
# ============================================================

st.sidebar.subheader("📄 HLD Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload AUTOSAR HLD PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.sidebar.button("Load HLD"):

        temp_pdf_path = "uploaded_hld.pdf"

        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        backend.load_hld_pdf(temp_pdf_path)

        st.session_state["hld_loaded"] = True
        st.sidebar.success("HLD loaded successfully!")


# ============================================================
# NAVIGATION
# ============================================================

section = st.sidebar.radio(
    "Select Feature",
    [
        "Ask HLD",
        "Component Analysis",
        "Dependency Analysis"
    ]
)


# ============================================================
# ASK HLD
# ============================================================

if section == "Ask HLD":

    st.header("🔎 Ask HLD")

    question = st.text_input(
        "Enter your question about the HLD:",
        placeholder="What does the Communication Manager do?"
    )

    if st.button("Ask HLD"):

        if question.strip():

            result = backend.ask_hld(question)

            st.subheader("Answer")

            st.write(result["answer"])

            st.subheader("Sources")

            for source in result["sources"]:

                st.write(
                    f"Page {source['page']} "
                    f"(Similarity: {source['score']})"
                )

            st.subheader("Similarity")

            st.write(round(result["score"], 3))

        else:

            st.warning("Please enter a question.")


# ============================================================
# COMPONENT ANALYSIS
# ============================================================

elif section == "Component Analysis":

    st.header("🧩 Component Analysis")

    component_names = [
        component["name"]
        for component in backend.components
    ]

    selected_component = st.selectbox(
        "Select an AUTOSAR component:",
        component_names
    )

    if st.button("Analyze Component"):

        analysis = backend.get_component_analysis(
            selected_component
        )

        st.subheader("Component")

        st.write(analysis["component"])

        st.subheader("Source Pages")

        st.write(analysis["source_pages"])

        st.subheader("Incoming Dependencies")

        if analysis["incoming"]:

            for dependency in analysis["incoming"]:

                st.write(
                    f"{dependency['source']} "
                    f"→ {dependency['relationship']} → "
                    f"{dependency['target']} "
                    f"(Page {dependency['page']})"
                )

        else:

            st.write("No incoming dependencies found.")

        st.subheader("Outgoing Dependencies")

        if analysis["outgoing"]:

            for dependency in analysis["outgoing"]:

                st.write(
                    f"{dependency['source']} "
                    f"→ {dependency['relationship']} → "
                    f"{dependency['target']} "
                    f"(Page {dependency['page']})"
                )

        else:

            st.write("No outgoing dependencies found.")

        st.subheader("Functional Flow")

        st.write(analysis["functional_flow"])


# ============================================================
# DEPENDENCY ANALYSIS
# ============================================================

elif section == "Dependency Analysis":

    st.header("🔗 Dependency Analysis")

    dependencies = backend.analyze_dependencies()

    st.write(
        f"Total Dependencies: {len(dependencies)}"
    )

    for i, dependency in enumerate(
        dependencies,
        start=1
    ):

        st.subheader(
            f"Dependency {i}"
        )

        st.write(
            f"{dependency['source']} "
            f"→ {dependency['relationship']} → "
            f"{dependency['target']}"
        )

        st.caption(
            f"Source: Page {dependency['page']}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AUTOSAR HLD AI Assistant | 1-Week MVP"
)