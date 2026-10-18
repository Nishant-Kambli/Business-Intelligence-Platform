"""Training analytics page enabled when training fields are present."""

import streamlit as st

from ai_bi.module_analytics import render_module
from ai_bi.utils import setup_page

setup_page("Training Analytics")
render_module("training", "Review training participation, outcomes, duration, and costs recorded in this dataset.")
