import pathlib

import pandas as pd
import yaml

path = pathlib.Path(__file__).parent
run_folder = path / "runs"

for run in run_folder.iterdir():
    if run.is_dir():
        config_path = run / "config.yaml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Cargar el archivo CSV
        csv_file = run / "Predicted.csv"
        prompt = config["prompt_sd"]
        df = pd.read_csv(csv_file)
        print(f"Prompt: {prompt}")
        generos_espanol = {
            "Male": "Hombre",
            "Female": "Mujer",
            "Other": "Otro"
        }
        genders = ["Male","Female","Other"]
        print("Gender")
        gender_value_count = df['gender'].value_counts()
        for index, value in gender_value_count.items():
            print(f"({generos_espanol[index]}, {value})",end=" ")
        for gender in genders:
            if gender not in gender_value_count:
                print(f"({generos_espanol[gender]}, 0)",end=" ")
        print()
        print("Race")
        razas_espanol = {
            "Black": "Negro",
            "White": "Blanco",
            "Asian": "Asiatico",
            "Indian": "Indio",
            "Latin": "Latino",
            "Other": "Otro"
        }
        races = ["Black","White",'Asian','Indian','Latin','Other']
        race_value_count = df['race'].value_counts()
        for index, value in race_value_count.items():
            print(f"({razas_espanol[index]}, {value})",end=" ")
        
        for race in races:
            if race not in race_value_count:
                print(f"({razas_espanol[race]}, 0)", end=" ")

        print("\n\n\n")

