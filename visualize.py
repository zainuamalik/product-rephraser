# FOR CUSTOM PORT: --server.port {custom_port}
# FOR CUSTOM BASEURL PATH: --server.baseUrlPath={custom_base_url_path}


import pandas as pd
import streamlit as st
import textwrap
import os
import sys
import argparse
import os


# Configuration
CSV_PATH = os.environ.get(
    "CSV_PATH",
    "data/output/rephrased_products_enhanced.csv",
)
PRODUCTS_PER_PAGE = 5
HTML_HEIGHT = 550


def render_html_container(content):
    """Create a styled container for HTML content"""
    if not content or not content.strip():
        return "<p>No content</p>"

    return f"""
    <div style='
        background-color: white; 
        color: black; 
        padding: 10px;
        height: {HTML_HEIGHT}px; 
        overflow-y: auto; 
        border: 1px solid #ccc; 
        border-radius: 5px;
        font-family: sans-serif;
    '>
        {content}
    </div>
    """


def main():
    st.set_page_config(layout="wide", page_title="Product Comparison")
    st.title("Product Description Comparison Tool")

    # Load data
    try:
        df = pd.read_csv(CSV_PATH)
        df = df.fillna("")
    except FileNotFoundError:
        st.error(f"Data file not found: {CSV_PATH}")
        return
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # Pagination setup
    total_products = len(df)
    total_pages = max(1, (total_products - 1) // PRODUCTS_PER_PAGE + 1)

    # Get current page from query params
    page = st.query_params.get("page", "1")
    try:
        page = max(1, min(int(page), total_pages))
    except ValueError:
        page = 1

    start_idx = (page - 1) * PRODUCTS_PER_PAGE
    end_idx = min(start_idx + PRODUCTS_PER_PAGE, total_products)
    current_page_df = df.iloc[start_idx:end_idx]

    # Display products
    for _, row in current_page_df.iterrows():
        st.divider()
        st.subheader(f"Product ID: {row.get('id', 'N/A')}")

        col_title1, col_title2 = st.columns(2)
        with col_title1:
            st.caption(f"**Original Title:**")
            st.text(textwrap.shorten(row.get("title", ""), width=300))

        with col_title2:
            st.caption(f"**Rephrased Title:**")
            st.text(textwrap.shorten(row.get("rephrased_title", ""), width=300))

        col1, col2 = st.columns(2)

        # Original description
        with col1:
            st.markdown("**Original Description**")
            st.components.v1.html(
                render_html_container(row.get("description", "")),
                height=HTML_HEIGHT + 20,
            )

        # Rephrased description
        with col2:
            st.markdown("**Rephrased Description**")
            st.components.v1.html(
                render_html_container(row.get("rephrased_description", "")),
                height=HTML_HEIGHT + 20,
            )

    # Pagination controls
    st.divider()
    # Updated to show both product range and page information
    st.caption(
        f"Showing products {start_idx + 1}-{end_idx} of {total_products} | Page {page} of {total_pages}"
    )

    col_prev, col_page, col_next = st.columns([1, 2, 1])

    with col_prev:
        if page > 1:
            if st.button("← Previous"):
                st.query_params = {"page": str(page - 1)}
                st.rerun()

    with col_page:
        new_page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=page,
            step=1,
            key="page_input",
            label_visibility="collapsed",
        )
        if new_page != page:
            st.query_params = {"page": str(new_page)}
            st.rerun()

    with col_next:
        if page < total_pages:
            if st.button("Next →"):
                st.query_params = {"page": str(page + 1)}
                st.rerun()


if __name__ == "__main__":
    main()
