# Stream Composition Analyzer

This application allows you to analyze chemical stream compositions by entering data in a table or uploading a spreadsheet.

## Features

- Enter chemical data directly in an interactive table
- Upload data via CSV or Excel file
- Process chemical composition data and get analysis results
- Supports different chemical types: Interferent, Matrix, and Measurement

## How to Use

1. Enter chemical data in the table or upload a spreadsheet
2. Click "Submit" to process the data
3. View the analysis results

## File Format

For uploaded spreadsheets, ensure they contain these columns:
- Component Name
- Nominal Concentration
- Maximum Concentration
- Type (Interferent, Matrix, or Measurement)
```

## Step 2: Create a Hugging Face Space

1. Go to [Hugging Face](https://huggingface.co/) and sign in or create an account
2. Click on your profile picture and select "New Space"
3. Fill in the following details:
   - Owner: Your username
   - Space name: "stream-composition-analyzer" (or your preferred name)
   - License: Choose appropriate license
   - SDK: Select "Gradio"
   - Space hardware: CPU (default)
   - Make it public

## Step 3: Upload your files to the Space

You have two options:

### Option 1: Using the web interface

1. After creating the Space, click "Files" in the repository
2. Upload the files you created (app.py, requirements.txt, README.md)
3. Also upload the necessary data directories (mw, output) if needed

### Option 2: Using Git

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/stream-composition-analyzer
```

```bash
cd stream-composition-analyzer
```

```bash
cp /path/to/your/app.py .
```

```bash
cp /path/to/your/requirements.txt .
```

```bash
cp /path/to/your/README.md .
```

```bash
git add .
```

```bash
git commit -m "Initial commit of Stream Composition Analyzer"
```

```bash
git push