import os
import argparse
import pandas as pd
import streamlit as st
import textwrap
from dotenv import load_dotenv

load_dotenv()

# Configuration
CSV_PATH = os.environ.get(
    "CSV_PATH",
    "data/output/rephrased_products_enhanced.csv",
)
PRODUCTS_PER_PAGE = 50


def check_password():
    def password_entered():
        if st.session_state["username"] == os.environ.get(
            "BASIC_AUTH_USERNAME"
        ) and st.session_state["password"] == os.environ.get("BASIC_AUTH_PASSWORD"):
            st.session_state["authenticated"] = True
            del st.session_state["password"]
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.text_input("Username", key="username", on_change=password_entered)
        st.text_input(
            "Password", type="password", key="password", on_change=password_entered
        )
        st.stop()
    elif not st.session_state["authenticated"]:
        st.error("Invalid username or password")
        st.text_input("Username", key="username", on_change=password_entered)
        st.text_input(
            "Password", type="password", key="password", on_change=password_entered
        )
        st.stop()


def render_html_container(content: str) -> str:
    """Wrap content in a styled div that expands to fit its children."""
    return f"""
    <div style='
        background-color: white;
        color: black;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        font-family: sans-serif;
    '>
        {content}
    </div>
    """


def main():
    # Allow overriding port/base URL via CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--server.port", dest="custom_port", type=int)
    parser.add_argument("--server.baseUrlPath", dest="custom_base_url_path", type=str)
    args, _ = parser.parse_known_args()
    if args.custom_port:
        os.environ["STREAMLIT_SERVER_PORT"] = str(args.custom_port)
    if args.custom_base_url_path:
        os.environ["STREAMLIT_SERVER_BASEURL_PATH"] = args.custom_base_url_path

    check_password()
    st.set_page_config(layout="wide", page_title="Product Comparison")
    st.title("Product Description Comparison Tool")

    # Load data
    try:
        df = pd.read_csv(CSV_PATH).fillna("")
    except FileNotFoundError:
        st.error(f"Data file not found: {CSV_PATH}")
        return
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Pagination setup
    total_products = len(df)
    total_pages = max(1, (total_products - 1) // PRODUCTS_PER_PAGE + 1)
    page_str = st.query_params.get("page", ["1"])[0]
    try:
        page = max(1, min(int(page_str), total_pages))
    except ValueError:
        page = 1

    start_idx = (page - 1) * PRODUCTS_PER_PAGE
    end_idx = min(start_idx + PRODUCTS_PER_PAGE, total_products)
    current_page_df = df.iloc[start_idx:end_idx]

    # Display products
    for _, row in current_page_df.iterrows():
        st.divider()
        st.subheader(f"Product ID: {row.get('id', 'N/A')}")

        # Titles
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.caption("**Original Title:**")
            st.text(textwrap.shorten(row["title"], width=300))
        with col_t2:
            st.caption("**Rephrased Title:**")
            st.text(textwrap.shorten(row["rephrased_title"], width=300))

        # Descriptions
        col1, col2 = st.columns(2)

        # Original description
        desc1 = row.get("description", "").strip()
        if desc1:
            with col1:
                st.markdown("**Original Description**")
                st.markdown(
                    render_html_container(desc1),
                    unsafe_allow_html=True,
                )

        # Rephrased description
        desc2 = row.get("rephrased_description", "").strip()
        if desc2:
            with col2:
                st.markdown("**Rephrased Description**")
                st.markdown(
                    render_html_container(desc2),
                    unsafe_allow_html=True,
                )

    # Pagination controls
    st.divider()
    st.caption(
        f"Showing products {start_idx + 1}-{end_idx} of {total_products} | Page {page} of {total_pages}"
    )
    col_prev, col_page, col_next = st.columns([1, 2, 1])

    with col_prev:
        if page > 1 and st.button("← Previous"):
            st.query_params = {"page": str(page - 1)}
            st.experimental_rerun()

    with col_page:
        new_page = st.number_input(
            "Page",  # non-empty label, but hidden
            min_value=1,
            max_value=total_pages,
            value=page,
            step=1,
            key="page_input",
            label_visibility="collapsed",
        )
        if new_page != page:
            st.query_params = {"page": str(new_page)}
            st.experimental_rerun()

    with col_next:
        if page < total_pages and st.button("Next →"):
            st.query_params = {"page": str(page + 1)}
            st.experimental_rerun()


if __name__ == "__main__":
    main()
