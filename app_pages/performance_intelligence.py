"""Performance analytics page enabled when performance fields are present."""

import streamlit as st

from ai_bi.module_analytics import render_module
from ai_bi.utils import setup_page

setup_page("Performance Intelligence")
render_module("performance", "Explore recorded employee ratings and performance scores by available workforce groups.")
