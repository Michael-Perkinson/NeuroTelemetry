# Pressure Analysis — How to Run

This guide shows you how to run the project on your computer, even if you’ve never used Python before.

- You’ll install Python.
- You’ll create a **virtual environment (venv)** so everything stays clean.
- You’ll install the project’s required packages.
- You’ll run the program.

---

## Windows (PC)

### 1) Install Python (one time)
1. Open: https://www.python.org/downloads/windows/
2. Click **Download Python 3.x.x**.
3. **Important:** On the first installer screen, tick **“Add Python to PATH”**.
4. Click **Install Now** and finish.

### 2) Get the project folder
- On GitHub: click **Code → Download ZIP**, then **Extract All…**.
- Remember where the extracted folder is (e.g. `C:\Users\You\Downloads\ProjectName`).

### 3) Open Command Prompt in the project folder
1. Press **Win + R**, type `cmd`, press **Enter**.
2. Change directory to your project folder (tip: type `cd ` then drag the folder into the window):
   ```
   cd "C:\path\to\your\ProjectName"
   ```

### 4) Create a virtual environment (venv)
   ```
   python -m venv .venv
   ```

### 5) Activate the virtual environment
   ```
   .venv\Scripts\activate.bat
   ```
   You should see `(.venv)` at the start of the line.

### 6) Upgrade pip and install requirements
   ```
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

### 7) Run the program
   ```
   python main.py
   ```

### 8) Next time you come back
   ```
   cd "C:\path\to\your\ProjectName"
   .venv\Scripts\activate.bat
   python main.py
   ```
   To stop using the virtual environment:
   ```
   deactivate
   ```

---

## Mac (macOS)

### 1) Install Python (one time)
1. Open: https://www.python.org/downloads/macos/
2. Click **Download Python 3.x.x**.
3. Open the downloaded `.pkg` and follow the prompts.  
   > If macOS blocks it, go to **System Settings → Privacy & Security**, click **Open Anyway**.

### 2) Get the project folder
- On GitHub: click **Code → Download ZIP**, then double-click the ZIP to extract.
- Remember where the extracted folder is (e.g. `/Users/YourName/Downloads/ProjectName`).

### 3) Open Terminal in the project folder
1. Press **Command + Space**, type **Terminal**, press **Enter**.
2. Change directory to your project (tip: type `cd ` then drag the folder into Terminal):
   ```
   cd /path/to/your/ProjectName
   ```

### 4) Create a virtual environment (venv)
   ```
   python3 -m venv venv
   ```

### 5) Activate the virtual environment
   ```
   source venv/bin/activate
   ```
   You should see `(venv)` at the start of the line.

### 6) Upgrade pip and install requirements
   ```
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   *(Inside the venv, `python` points to the right place.)*

### 7) Run the program
   ```
   python3 main.py
   ```

### 8) Next time you come back
   ```
   cd /path/to/your/ProjectName
   source venv/bin/activate
   python3 main.py
   ```
   To stop using the virtual environment:
   ```
   deactivate
   ```

---

## Troubleshooting

- **`pip` not found**  
  Use `python -m pip ...` (shown above) instead of `pip`.

- **`python` not found on Mac**  
  Use `python3` to create the venv. After activation, `python` works inside the venv.

- **`No such file or directory: .venv/bin/activate`**  
  You didn’t create the venv yet or you’re in the wrong folder. Run the venv creation step again and confirm you’re inside the project folder.

- **Want to remove the venv and start over?**  
  Close the terminal, delete the `.venv` folder, then redo steps 4–6.
