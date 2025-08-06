#!/bin/bash
pip install --no-cache-dir -r requirements.txt
streamlit run app.py --server.port=$PORT --server.address 0.0.0.0
