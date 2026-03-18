======================================================
TRAFFIC PREDICTION GNN - PROJECT SETUP & RUN GUIDE
======================================================

--- PREREQUISITES ---
Ensure you have Python 3.9, 3.10, or 3.11 installed on your system.
(You can verify this by running `python --version` or `python3 --version` in your terminal).

* Windows Users: Ensure you installed Python from the official Python.org installer (NOT MSYS2/MinGW) and checked the "Add python.exe to PATH" box during installation.

======================================================
PART 1A: ENVIRONMENT SETUP (WINDOWS - POWERSHELL)
======================================================
Open PowerShell, navigate to this project folder, and run the following commands sequentially:

1. Create a new virtual environment named '.venv':
   > python -m venv .venv

2. Activate the virtual environment:
   > .\.venv\Scripts\Activate.ps1
   (Note: If you get an Execution Policy error, run this first: `Set-ExecutionPolicy Unrestricted -Scope CurrentUser`)

3. Upgrade pip to the latest version:
   > python -m pip install --upgrade pip

4. Install standard project dependencies:
   > pip install -r requirements.txt

5. Install PyTorch:
   > pip install torch

6. Install PyTorch Geometric and Temporal:
   (We do this after PyTorch to prevent C++ build errors)
   > pip install torch_geometric torch_geometric_temporal

======================================================
PART 1B: ENVIRONMENT SETUP (LINUX & MACOS - BASH/ZSH)
======================================================
Open your terminal, navigate to this project folder, and run the following commands sequentially:

1. Create a new virtual environment named '.venv':
   > python3 -m venv .venv

2. Activate the virtual environment:
   > source .venv/bin/activate

3. Upgrade pip to the latest version:
   > pip install --upgrade pip

4. Install standard project dependencies:
   > pip install -r requirements.txt

5. Install PyTorch:
   > pip install torch

6. Install PyTorch Geometric and Temporal:
   > pip install torch_geometric torch_geometric_temporal

======================================================
PART 2: RUNNING THE PROJECT (ALL PLATFORMS)
======================================================
To run the full stack, you will need to open THREE separate terminal windows. 

*IMPORTANT:* Make sure you activate your virtual environment in the terminals running the Training and Backend!
(Windows: `.\.venv\Scripts\Activate.ps1` | Linux/Mac: `source .venv/bin/activate`)

--- TERMINAL 1: TRAINING THE AI ---
1. Ensure your CSV data files are in the same folder as the scripts.
2. Run the training script:
   > python train.py
3. Wait for the epochs to finish and the 'traffic_gnn_weights.pth' file to be generated.

--- TERMINAL 2: THE BACKEND API ---
1. Start the FastAPI server:
   > uvicorn main:app --reload
2. The server will host on http://127.0.0.1:8000

--- TERMINAL 3: THE FRONTEND MAP ---
1. Start a simple web server for your HTML file:
   > python -m http.server 8080
2. Open your web browser and navigate to: http://localhost:8080
3. Click two points on the map to trigger a prediction!
