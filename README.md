# AHHunter - Aspect Of The Flipper but make it cheaper
AHHunter is a script that helps with flipping items on the Hypixel Skyblock Auction House.

## Setup

### 1. Python Version
Ensure you have Python 3.8+ installed.

### 2. Dependencies
Install the required Python packages using pip and the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. API Key
This script requires a Hypixel API key to function.
1.  Obtain a key by running the `/api new` command in-game on the Hypixel server.
2.  Set the key as an environment variable named `HYPIXEL_API_KEY`.

On Windows (PowerShell):
```powershell
$env:HYPIXEL_API_KEY="your-key-here"
```

On Linux/macOS:
```bash
export HYPIXEL_API_KEY="your-key-here"
```

### 4. Data Files
The `reforges.csv` file contains a list of item reforges that are stripped from item names to normalize them for price analysis. This file should be in the same directory as the script.

## Running the script
Once the setup is complete, run the script from your terminal:
```bash
python AHHunter.py
```
The script will continuously scan the auction house, identify underpriced items using anomaly detection, and print potential flips to the console.

## Platform Compatibility
The script is cross-platform and has been updated to work on Windows, macOS, and Linux. System-specific features like notification sounds are handled gracefully.

