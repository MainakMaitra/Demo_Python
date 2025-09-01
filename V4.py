from IPython.display import display, HTML
import pandas as pd

def fix_table_display():
    styles = """
    <style>
    table.dataframe td, table.dataframe th {
        white-space: initial !important;
        word-wrap: break-word;
    }
    div.output_scroll {
        overflow-x: auto !important;
    }
    </style>
    """
    display(HTML(styles))

# Apply CSS fix
fix_table_display()
