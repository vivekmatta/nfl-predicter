# Kaggle API Setup Guide

The "Refresh Model" button can download the latest data from Kaggle. To use this feature, you need to set up Kaggle API authentication.

## Quick Setup (Recommended - API Token Method)

### Step 1: Get Your Kaggle API Token

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Scroll down to the **"API"** section
3. Click **"Create New Token"**
4. A modal will appear with your API token (starts with `KGAT_`)
5. **Copy the token immediately** - you won't be able to see it again!

### Step 2: Set the Environment Variable

**On Windows (PowerShell):**
```powershell
$env:KAGGLE_API_TOKEN="KGAT_your_token_here"
```

**On Windows (Command Prompt):**
```cmd
set KAGGLE_API_TOKEN=KGAT_your_token_here
```

**On Linux/Mac:**
```bash
export KAGGLE_API_TOKEN=KGAT_your_token_here
```

**To make it permanent (Linux/Mac):**
Add the export line to your `~/.bashrc` or `~/.zshrc` file:
```bash
echo 'export KAGGLE_API_TOKEN=KGAT_your_token_here' >> ~/.bashrc
source ~/.bashrc
```

**To make it permanent (Windows):**
1. Open System Properties → Environment Variables
2. Add new User Variable:
   - Name: `KAGGLE_API_TOKEN`
   - Value: `KGAT_your_token_here`

### Step 3: Restart Your Flask Server

After setting the environment variable, restart your Flask backend:
```bash
python app.py
```

## Alternative Method: kaggle.json File (Legacy)

If you prefer the old method using a file:

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Scroll to "API" section and click "Create New Token"
3. Download `kaggle.json` file
4. Place it in:
   - **Windows**: `C:\Users\<your-username>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
5. Set permissions (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Alternative: Use Local CSV File

If you don't want to set up Kaggle authentication, you can:

1. Download the dataset manually from [Kaggle](https://www.kaggle.com/datasets/tonycorona/nfl-spreadspoke-scores)
2. Place `spreadspoke_scores.csv` in the project root directory
3. The "Refresh Model" button will automatically use the local file if Kaggle fails

## Testing

After setting up, click the "🔄 Refresh Model" button. If successful, you'll see:
- ✅ Model refreshed successfully from kaggle

If there's an error, check:
- The `KAGGLE_API_TOKEN` environment variable is set correctly
- You've restarted the Flask server after setting the variable
- You have internet connection
- The dataset is publicly accessible

## Troubleshooting

**"403" or "Permission denied" error:**
- Make sure `KAGGLE_API_TOKEN` is set correctly
- Restart your Flask server after setting the environment variable
- Check that the token hasn't expired (create a new one if needed)

**"401 Unauthorized" error:**
- Your API token might be invalid or expired
- Create a new token from Kaggle settings
- Make sure you copied the entire token (it's long!)

**"Dataset not found":**
- The dataset URL might have changed
- Check that `tonycorona/nfl-spreadspoke-scores` is still available on Kaggle

**Environment variable not working:**
- Make sure you set it in the same terminal/command prompt where you run `python app.py`
- On Windows, you may need to restart your terminal after setting the variable
- Try setting it directly in your terminal before running the Flask server

## Quick Test

To verify your token is set correctly, run this in your terminal:

**Windows:**
```cmd
echo %KAGGLE_API_TOKEN%
```

**Linux/Mac:**
```bash
echo $KAGGLE_API_TOKEN
```

You should see your token (starting with `KGAT_`). If you see nothing, the variable isn't set.
