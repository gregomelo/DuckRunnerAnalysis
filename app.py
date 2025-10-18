"""
Main entry point of the DuckRunner project.

This module initializes the Streamlit interface and displays the
welcome message. It will later be expanded to include the upload
of FIT files, data processing, and the rendering of interactive
performance charts.

Functions
---------
main()
    Launches the Streamlit application and renders the initial
    user interface.
"""

import streamlit as st


def main() -> None:
    """
    Run the main DuckRunner application.

    This function serves as the entry point for the Streamlit app.
    For now, it only displays a greeting message.

    Returns
    -------
    None
        This function does not return any value. It renders content
        directly to the Streamlit UI.
    """
    st.write("Hi, DuckRunner")


if __name__ == "__main__":
    main()
