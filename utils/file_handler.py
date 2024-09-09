import os
import pandas as pd
import csv
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

    def read_input_file(self, file_path=None):
        if file_path is None:
            file_path = self.get_input_file()

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, dtype=str)
            sheets = None
        else:
            xl = pd.ExcelFile(file_path)
            sheets = xl.sheet_names
            print("Available sheets:")
            for i, sheet in enumerate(sheets):
                print(f"{i + 1}. {sheet}")
            sheet_index = int(input("Enter the number of the sheet to use: ")) - 1
            df = xl.parse(sheets[sheet_index], dtype=str)

        print("\nAvailable columns:")
        for i, column in enumerate(df.columns):
            print(f"{i + 1}. {column}")
        column_index = int(input("Enter the number of the column to use: ")) - 1

        chosen_column = df.columns[column_index]
        
        df[chosen_column] = df[chosen_column].astype(str)
        df = df.dropna(subset=[chosen_column])
        df = df[df[chosen_column].str.strip() != '']

        logger.info(f"Read {len(df)} rows from {file_path}")
        logger.info(f"Chosen column: {chosen_column}")

        return df[chosen_column].tolist(), chosen_column, sheets[sheet_index] if sheets else None

    def read_input_file(self, file_path):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, dtype=str)
            sheets = None
        else:
            xl = pd.ExcelFile(file_path)
            sheets = xl.sheet_names
            print("Available sheets:")
            for i, sheet in enumerate(sheets):
                print(f"{i + 1}. {sheet}")
            sheet_index = int(input("Enter the number of the sheet to use: ")) - 1
            df = xl.parse(sheets[sheet_index], dtype=str)

        print("\nAvailable columns:")
        for i, column in enumerate(df.columns):
            print(f"{i + 1}. {column}")
        column_index = int(input("Enter the number of the column to use: ")) - 1

        chosen_column = df.columns[column_index]
        
        df[chosen_column] = df[chosen_column].astype(str)
        df = df.dropna(subset=[chosen_column])
        df = df[df[chosen_column].str.strip() != '']

        logger.info(f"Read {len(df)} rows from {file_path}")
        logger.info(f"Chosen column: {chosen_column}")

        return df[chosen_column].tolist(), chosen_column, sheets[sheet_index] if sheets else None

    def write_results(self, items, results, chosen_column):
        output_file_path = os.path.join(self.output_path, 'classification_results.csv')
        logger.info(f"Writing results to {output_file_path}")

        try:
            with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Item', 'Primary_Classification', 'Overall_Classification', 'Reasoning', 'Confidence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for item, result in zip(items, results):
                    writer.writerow({
                        'Item': item,
                        'Primary_Classification': result['primary_classification'],
                        'Overall_Classification': result['classification'],
                        'Reasoning': result['reasoning'],
                        'Confidence': result['confidence']
                    })

            logger.info(f"Results successfully written to {output_file_path}")
        except Exception as e:
            logger.error(f"Error writing results to CSV: {str(e)}")
            raise

file_handler = FileHandler()