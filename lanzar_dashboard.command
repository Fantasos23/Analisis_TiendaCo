#!/bin/zsh
cd "$(dirname "$0")"

echo "🔄 Sincronizando repositorio con GitHub..."
git pull origin main
git add .
git commit -m "Auto-update desde Mac: $(date '+%Y-%m-%d %H:%M:%S')"
git push origin main

echo "🚀 Iniciando Dashboard de Streamlit..."
streamlit run app.py