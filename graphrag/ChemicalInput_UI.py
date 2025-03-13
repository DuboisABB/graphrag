import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from graphrag.cli.query import run_local_search
from pathlib import Path

class StreamCompositionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stream Composition Analyzer")
        self.root.geometry("900x600")
        self.root.minsize(750, 400)
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TLabel", font=("Arial", 10))
        self.style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        self.style.configure("TableHeader.TLabel", font=("Arial", 10, "bold"))
        
        # Main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Stream Composition Entry", style="Header.TLabel")
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_label = ttk.Label(main_frame, text="Enter the chemicals in your stream composition below.")
        desc_label.pack(pady=(0, 10), anchor="w")
        
        # Container for scrolling
        container = ttk.Frame(main_frame)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas with scrollbar
        self.canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Table frame
        self.table_frame = ttk.Frame(self.scrollable_frame)
        self.table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Table headers
        self.create_table_headers()
        
        # List to store chemical rows data
        self.chemicals = []
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Add chemical button
        add_btn = ttk.Button(button_frame, text="Add Chemical", command=self.add_chemical)
        add_btn.pack(side=tk.LEFT, padx=5)
        
        # Submit button
        submit_btn = ttk.Button(button_frame, text="Submit", command=self.submit)
        submit_btn.pack(side=tk.RIGHT, padx=5)
        
        # Result frame
        result_frame = ttk.Frame(main_frame, padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Result label
        result_label = ttk.Label(result_frame, text="Generated Output:")
        result_label.pack(anchor="w", pady=(0, 5))
        
        # Result text
        self.result_text = tk.Text(result_frame, height=5, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Row counter
        self.row_count = 1  # Start after header row
        
        # Add first chemical row by default
        self.add_chemical()
    
    def create_table_headers(self):
        """Create the table headers"""
        headers = ["Component Name", "Nominal Concentration", "Maximum Concentration", "Type", "Actions"]
        
        for col, header in enumerate(headers):
            label = ttk.Label(self.table_frame, text=header, style="TableHeader.TLabel")
            label.grid(row=0, column=col, padx=5, pady=5, sticky="w")
            
        # Configure columns to expand properly
        self.table_frame.columnconfigure(0, weight=3)  # Component name gets more space
        self.table_frame.columnconfigure(1, weight=2)
        self.table_frame.columnconfigure(2, weight=2)
        self.table_frame.columnconfigure(3, weight=2)
        self.table_frame.columnconfigure(4, weight=1)
    
    def add_chemical(self):
        """Add a new chemical row to the table"""
        row_idx = self.row_count
        
        # Component name
        name_var = tk.StringVar()
        name_entry = ttk.Entry(self.table_frame, textvariable=name_var, width=30)
        name_entry.grid(row=row_idx, column=0, padx=5, pady=5, sticky="ew")
        
        # Nominal concentration
        nominal_var = tk.StringVar()
        nominal_entry = ttk.Entry(self.table_frame, textvariable=nominal_var, width=15)
        nominal_entry.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        
        # Maximum concentration
        max_var = tk.StringVar()
        max_entry = ttk.Entry(self.table_frame, textvariable=max_var, width=15)
        max_entry.grid(row=row_idx, column=2, padx=5, pady=5, sticky="ew")
        
        # Chemical type dropdown
        type_var = tk.StringVar(value="Interferent")
        type_combo = ttk.Combobox(self.table_frame, textvariable=type_var, 
                                  values=["Interferent", "Matrix", "Measurement"],
                                  state="readonly", width=15)
        type_combo.grid(row=row_idx, column=3, padx=5, pady=5, sticky="ew")
        
        # Remove button
        remove_btn = ttk.Button(self.table_frame, text="Remove", 
                               command=lambda idx=row_idx: self.remove_chemical(idx))
        remove_btn.grid(row=row_idx, column=4, padx=5, pady=5, sticky="w")
        
        # Store row data
        self.chemicals.append({
            'row': row_idx,
            'name_var': name_var,
            'nominal_var': nominal_var,
            'max_var': max_var,
            'type_var': type_var,
            'widgets': [name_entry, nominal_entry, max_entry, type_combo, remove_btn]
        })
        
        self.row_count += 1
        
        # Adjust the canvas scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Scroll to the bottom
        self.canvas.yview_moveto(1.0)
    
    def remove_chemical(self, row_idx):
        """Remove a chemical row by index"""
        if len(self.chemicals) <= 1:
            messagebox.showwarning("Warning", "At least one chemical is required.")
            return
        
        # Find the chemical by row index
        to_remove = None
        for idx, chem in enumerate(self.chemicals):
            if chem['row'] == row_idx:
                to_remove = idx
                break
        
        if to_remove is not None:
            # Remove widgets from the grid
            for widget in self.chemicals[to_remove]['widgets']:
                widget.destroy()
            
            # Remove from our list
            self.chemicals.pop(to_remove)
            
            # Adjust the canvas scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def validate(self):
        """Validate that all required fields are filled"""
        for idx, chem in enumerate(self.chemicals):
            if not chem['name_var'].get().strip():
                messagebox.showerror("Error", f"Please enter a Component Name for row {idx+1}")
                return False
            if not chem['nominal_var'].get().strip():
                messagebox.showerror("Error", f"Please enter a Nominal Concentration for row {idx+1}")
                return False
            if not chem['max_var'].get().strip():
                messagebox.showerror("Error", f"Please enter a Maximum Concentration for row {idx+1}")
                return False
        return True
    
    def submit(self):
        """Process the submitted data"""
        # Validate inputs
        if not self.validate():
            return
        
        # Generate output string
        output = "Chemicals: "
        for i, chem in enumerate(self.chemicals):
            if i > 0:
                output += ", "
            output += f"-Name:{chem['name_var'].get()}, " \
                     f"Type:{chem['type_var'].get()}, " \
                     f"Nominal_Concentration:{chem['nominal_var'].get()}, " \
                     f"Max_Concentration:{chem['max_var'].get()}"
        
        # Display the result
        self.result_text.delete(1.0, tk.END)
        

        response, context_data = run_local_search(
                config_filepath=Path("./mw/settings.yaml"), # or None to use default
                data_dir=Path("../mw2/output"),
                root_dir=Path("./mw"),                    # or None to use config default
                community_level=2,                            # Adjust as needed
                query= output,
                response_type="default",                      # If applicable
                streaming=False                                # If applicable
            )

        self.result_text.insert(tk.END, f"\n{response}")

        # Show success message
        messagebox.showinfo("Success", "Stream composition processed successfully!")


if __name__ == "__main__":
    root = tk.Tk()
    app = StreamCompositionApp(root)
    root.mainloop()
