from pprint import pprint
from peliculas.etl_characteristics import load_data, process_data_characteristics, process_data_characteristics
from peliculas import consts
from peliculas.etl_description import load_data, process_data_description

def main() -> None:
    data= load_data(consts.DATA_PATH / "netflix_titles.csv")
    data_char= process_data_characteristics(data) 
    data_desc = process_data_description(data)
