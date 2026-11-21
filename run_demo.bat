@echo off
REM Colours every image in D:\surgseg\shots and writes results to D:\surgseg\demo_out
cd /d D:\surgseg
.venv\Scripts\python -m surgseg.tools.demo_images --model models\surgseg.onnx --images shots --out demo_out --device-id 1
pause
