@echo off
REM Starts the surgseg model server for Unity (keep this window open while Unity runs).
cd /d D:\surgseg
.venv\Scripts\python -m surgseg.runtime.server --model models\surgseg.onnx --procedure blood_suction --port 5555 --device-id 1 --input-hw 512 640
pause
