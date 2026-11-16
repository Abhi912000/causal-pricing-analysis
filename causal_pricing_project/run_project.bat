
@echo off
cd /d "C:\Users\HP\Downloads\causal_pricing_project_vscode"

echo Starting Causal Pricing Analytics...

start "Causal Pricing Dashboard" /min ".venv\Scripts\python.exe" -m streamlit run "causal_pricing_project\app.py" --server.headless true

timeout /t 3 /nobreak >nul

start "" code "C:\Users\HP\Downloads\causal_pricing_project_vscode"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$url='http://localhost:8501'; for ($i=0; $i -lt 45; $i++) { try { $r=Invoke-WebRequest -Uri $url -TimeoutSec 2; if ($r.StatusCode -eq 200) { Start-Process $url; exit 0 } } catch {}; Start-Sleep -Seconds 1 }; Write-Host 'App did not start. Check the Streamlit terminal.'; pause"


