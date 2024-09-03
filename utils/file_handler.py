import os
import pandas as pd
from utils.logger import logger
from utils.config_loader import config

class FileHandler:
    def __init__(self):
        self.input_path = config['input_data_path']
        self.output_path = config['output_data_path']

    def get_input_file(self):
        files = os.listdir(self.input_path)
        csv_files = [f for f in files if f.endswith('.csv')]
        excel_files = [f for f in files if f.endswith(('.xls', '.xlsx'))]

        if not csv_files and not excel_files:
            logger.error("No CSV or Excel files found in the input directory.")
            raise FileNotFoundError("No input files found.")

        if len(csv_files) + len(excel_files) > 1:
            logger.error("Multiple input files found. Please ensure only one input file is present.")
            raise ValueError("Multiple input files found.")

        input_file = csv_files[0] if csv_files else excel_files[0]
        return os.path.join(self.input_path, input_file)

    def read_input_file(self, file_path):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            sheets = None
        else:
            xl = pd.ExcelFile(file_path)
            sheets = xl.sheet_names
            print("Available sheets:")
            for i, sheet in enumerate(sheets):
                print(f"{i + 1}. {sheet}")
            sheet_index = int(input("Enter the number of the sheet to use: ")) - 1
            df = xl.parse(sheets[sheet_index])

        print("\nAvailable columns:")
        for i, column in enumerate(df.columns):
            print(f"{i + 1}. {column}")
        column_index = int(input("Enter the number of the column to use: ")) - 1

        return df, df.columns[column_index], sheets[sheet_index] if sheets else None

    def write_output_file(self, input_df, results, input_column, file_name):
        output_df = input_df.copy()
        output_df['Similarity_Score'] = [result['similarity_score'] for result in results]
        output_df['Primary_Classification'] = [result['primary_classification'] for result in results]
        output_df['Reasoning'] = [result['reasoning'] for result in results]

        output_file_path = os.path.join(self.output_path, f"output_{file_name}")
        output_df.to_csv(output_file_path, index=False)
        logger.info(f"Output saved to {output_file_path}")

file_handler = FileHandler()