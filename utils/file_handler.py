import os
import pandas as pd
import csv
from utils.logger import get_logger
from utils.config_loader import config

logger = get_logger(__name__)

class FileHandler:
    def __init__(self):
        self.input_path = config.input_data_path
        self.output_path = config.output_data_path

    def get_input_file(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input directory not found: {self.input_path}")

        files = os.listdir(self.input_path)
        valid_extensions = ('.csv', '.xls', '.xlsx')
        valid_files = [f for f in files if f.endswith(valid_extensions)]

        if not valid_files:
            raise ValueError(f"No valid input files found. Supported formats: {', '.join(valid_extensions)}")

        if len(valid_files) > 1:
            logger.error("Multiple input files found. Please ensure only one input file is present.")
            raise ValueError("Multiple input files found.")

        input_file = valid_files[0]
        return os.path.join(self.input_path, input_file)

    def read_input_file(self, file_path):
        try:
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

            if df.empty:
                raise ValueError(f"The file {file_path} is empty.")

            print("\nAvailable columns:")
            for i, column in enumerate(df.columns):
                print(f"{i + 1}. {column}")
            column_index = int(input("Enter the number of the column to use: ")) - 1

            if column_index < 0 or column_index >= len(df.columns):
                raise ValueError("Invalid column index.")

            chosen_column = df.columns[column_index]
            
            df[chosen_column] = df[chosen_column].astype(str)
            df = df.dropna(subset=[chosen_column])
            df = df[df[chosen_column].str.strip() != '']

            if df.empty:
                raise ValueError(f"No valid data found in the chosen column '{chosen_column}'.")

            logger.info(f"Read {len(df)} rows from {file_path}")
            logger.info(f"Chosen column: {chosen_column}")

            return df[chosen_column].tolist(), chosen_column, sheets[sheet_index] if sheets else None

        except pd.errors.EmptyDataError:
            raise ValueError(f"The file {file_path} is empty.")
        except pd.errors.ParserError:
            raise ValueError(f"Unable to parse {file_path}. Please ensure it's a valid CSV or Excel file.")
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}")
            raise

    def write_results(self, items, results, chosen_column):
        if len(items) != len(results):
            raise ValueError("Mismatch between number of items and results.")

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