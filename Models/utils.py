"""
SMART COMPOST - MODEL PROJECT.

---  helper functions
---   Models/utils.py

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 22 Nov 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app
"""

# import
import pandas as pd
from sqlite3 import connect
from gc import collect
from time import time
from hashlib import sha256
import os


# variables
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


dataset_path = os.path.join(root_dir, "data", "smart_compost_dataset101.csv")
database_path = os.path.join(root_dir, "smart-compost.db")


def sqlite_bulk_insert( table: str, df: pd.DataFrame, database_path: str  = database_path):
    """Inserts a Pandas DataFrame into a SQLite database in bulk.

    Args:
        database_path (str): database path
        table (str): name of the table
        df (pd.DataFrame): pandas dataframe used to insert data chunks
    """
    with connect(database_path) as conn:
        conn.execute("PRAGMA journal_mode = OFF;")
        conn.execute("PRAGMA synchronous = OFF;")
        df.to_sql(table, conn, if_exists="append", index=False)



def Save_dataset_to_db(
    dataset_path: str = dataset_path,
    database_path: str = database_path,
    tablename: str = "smart_compost_dataset101",
    delimiter: str = ",",
):
    """Saves a large dataset to a SQLite database in chunks.

    Args:
        dataset_path (str):  Path to the CSV dataset.
        tablename (str):  Name of the SQLite table. Default is "smart_compost_dataset101".
        delimiter (str, optional): CSV delimiter. Defaults to ",".
    """

    CHUNK_SIZE = 100000
    ENCODING = "UTF-8"
    chunk_count = 1

    for chunk in pd.read_csv(
        dataset_path,
        chunksize=CHUNK_SIZE,
        sep=delimiter,
        on_bad_lines="skip",
        encoding=ENCODING,
    ):
        start = time()

        # Insert chunk into the database
        sqlite_bulk_insert(database_path, tablename, chunk)

        # Cleanup memory
        del chunk
        collect()

        print(f"Chunk {chunk_count} inserted in {time() - start:.2f} seconds")
        chunk_count += 1



def calculate_dataset_hash(dataset_path: str = dataset_path, delimiter: str = ",", encoding: str = "UTF-8") -> str:
    """
    Calculate a hash for the dataset to determine if it's the same as the one saved in the database.

     Args:
        dataset_path: Path to the CSV dataset.
        delimiter: CSV delimiter. Default is ",".
        encoding: Encoding of the CSV file. Default is "UTF-8".

    Returns:
        A SHA-256 hash of the dataset as a string.
    """
    hasher = sha256()
    with open(dataset_path, "rb") as file:
        while chunk := file.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()



def check_table_exists_and_same(database_path: str = database_path, tablename: str, dataset_hash: str) -> bool:
    """
    Check if a table exists in the database and if its hash matches the dataset hash.

     Args:
         database_path: Path to the SQLite database.
         tablename: Name of the table to check.
         dataset_hash: Hash of the dataset.

    Returns:
        True if the table exists and has the same hash, False otherwise.
    """
    with connect(database_path) as conn:
        cursor = conn.cursor()
        # Check if the table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (tablename,),
        )
        if cursor.fetchone():
            # Check if the hash matches
            cursor.execute(f"SELECT dataset_hash FROM {tablename} LIMIT 1;")
            result = cursor.fetchone()
            if result and result[0] == dataset_hash:
                return True
    return False



def save_dataset_with_incremented_table(
    dataset_path: str = dataset_path,
    database_path: str = database_path,
    tablename: str,
    delimiter: str = ",",
    encoding: str = "UTF-8",
):
    """
    Saves a dataset to the database. Skips if the dataset is already saved.
    If not, creates a new table with an incremented name.

    Args:
        dataset_path: Path to the CSV dataset.
        database_path: Path to the SQLite database.
        tablename: Base name for the table.
        delimiter: CSV delimiter (default is ",").
        encoding: Encoding of the CSV file (default is "UTF-8").
    """
    # Calculate the hash of the dataset
    dataset_hash = calculate_dataset_hash(dataset_path, delimiter, encoding)

    # Check if the table exists and the dataset matches
    if check_table_exists_and_same(database_path, tablename, dataset_hash):
        print(f"Dataset is already saved in the table '{tablename}'. Skipping...")
        return

    # If not, increment the table name
    new_tablename = tablename
    with connect(database_path) as conn:
        cursor = conn.cursor()
        while True:
            # Check if the table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (new_tablename,),
            )
            if not cursor.fetchone():
                break  # Table name is available
            # Increment table name (e.g., "tablename102")
            base_name, number = new_tablename.rstrip("0123456789"), new_tablename[len(tablename):]
            new_tablename = f"{base_name}{int(number or 101) + 1}"

    # Save the dataset to the new table
    print(f"Saving dataset to new table '{new_tablename}'...")
    for chunk in pd.read_csv(
        dataset_path,
        chunksize=100000,
        sep=delimiter,
        on_bad_lines="skip",
        encoding=encoding,
    ):
        chunk["dataset_hash"] = dataset_hash  # Store the dataset hash
        chunk.to_sql(new_tablename, connect(database_path), if_exists="append", index=False)

    print(f"Dataset saved to table '{new_tablename}'.")
