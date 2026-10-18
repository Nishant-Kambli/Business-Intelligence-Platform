"""Recruitment analytics page enabled when recruitment fields are present."""

import streamlit as st

from ai_bi.module_analytics import render_module
from ai_bi.utils import setup_page

setup_page("Recruitment Analytics")
render_module("recruitment", "Explore applications, hiring status, and sourcing records available in this dataset.")
