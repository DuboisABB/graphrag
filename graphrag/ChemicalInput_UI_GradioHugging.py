import gradio as gr
import pandas as pd
import numpy as np
from graphrag.cli.query_MW import run_local_search
from pathlib import Path

def run_search(df):
    """Process the chemicals data and run the search"""
    # Filter out rows with empty component names
    df_filtered = df[df['Component Name'].astype(str).str.strip() != '']
    
    if df_filtered.empty:
        return "Please add at least one chemical component with a name."
    
    # Generate output string
    output = "Chemicals: "
    for i, row in df_filtered.iterrows():
        if i > 0:
            output += ", "
        output += f"-Name:{row['Component Name']}, " \
                 f"Type:{row['Type']}, " \
                 f"Nominal_Concentration:{row['Nominal Concentration']}, " \
                 f"Max_Concentration:{row['Maximum Concentration']}"
    
    try:
        response, _ = run_local_search(
            config_filepath=Path("./mw/settings.yaml"),
            data_dir=Path("./output"),
            root_dir=Path("./mw"),
            community_level=2,
            query=output,
            response_type="default",
            streaming=False
        )
        return response
    except Exception as e:
        return f"Error processing request: {str(e)}"

def process_file(file):
    """Process uploaded CSV/Excel file and return as DataFrame"""
    file_path = file.name
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        return pd.DataFrame({
            'Component Name': [''],
            'Nominal Concentration': [''],
            'Maximum Concentration': [''],
            'Type': ['Interferent']
        })

def create_app():
    with gr.Blocks(title="Stream Composition Analyzer") as app:
        gr.Markdown("# Stream Composition Analyzer")
        gr.Markdown("Enter chemicals directly in the table or upload a spreadsheet.")
        
        # Initial dataframe with example data
        initial_df = pd.DataFrame({
            'Component Name': ['', '', ''],
            'Nominal Concentration': ['', '', ''],
            'Maximum Concentration': ['', '', ''],
            'Type': ['Interferent', 'Interferent', 'Interferent']
        })
        
        # Option 1: Interactive table (with improved copy/paste)
        df = gr.Dataframe(
            value=initial_df,
            interactive=True,
            elem_id="chemical_table",
            wrap=True
        )
        
        # Option 2: File upload for spreadsheet data
        with gr.Row():
            file_input = gr.File(label="Or upload CSV/Excel file")
        
        with gr.Row():
            load_btn = gr.Button("Load From File")
            submit_btn = gr.Button("Submit", variant="primary")
        
        result = gr.Textbox(label="Results", lines=10)
        
        # Connect buttons to functions
        load_btn.click(fn=process_file, inputs=[file_input], outputs=[df])
        submit_btn.click(fn=run_search, inputs=[df], outputs=[result])
        
        gr.Markdown("### Instructions:")
        gr.Markdown("""
        1. **Option A**: Enter data directly in the table
           - For multi-cell paste, try using the file upload option instead
        
        2. **Option B**: Upload a spreadsheet with columns:
           - Component Name, Nominal Concentration, Maximum Concentration, Type
           - Valid Types: Interferent, Matrix, Measurement
        
        3. Click Submit when ready
        """)
    
    return app

# Create and launch the app
app = create_app()
app.launch(server_name="0.0.0.0", share=True)