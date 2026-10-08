from pprint import pprint
from peliculas.etl_characteristics import load_data, process_data
from peliculas import consts

def main() -> None:
    data= load_data(consts.DATA_PATH / "netflix_titles.csv")
    data= process_data(data) 
