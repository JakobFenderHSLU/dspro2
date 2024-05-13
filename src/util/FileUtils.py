import os
import pathlib
import sys
import threading
import pandas as pd

from src.util.AudioUtil import AudioUtil


def _verify_files(df, result, verbose: bool = False):
    for index, row in df.iterrows():
        file_path = pathlib.Path(row["file_path"])

        if not file_path.exists():
            result["not_exist"].append(index)

        if file_path.suffix != ".mp3":
            result["not_mp3"].append(index)

        try:
            # Note (Jakob): This outputs a lot of useless information and spams the console.
            # I have not found a way to suppress it.
            AudioUtil.open(file_path)
        except Exception as e:
            result["corrupted"].append(index)
            if verbose:
                print(f"Corrupted: {file_path} - {e}")


def verify_data(df: pd.DataFrame, n_threads: int = 16, verbose: bool = False):
    """
    Verify that the files in the DataFrame exist and the Audio files are not corrupted.
    :param df: DataFrame with the file paths
    :param n_threads: Number of threads to use
    :param verbose: If True, print additional information
    :return: not_exist, not_mp3, corrupted
    """

    result = {
        "not_exist": [],
        "not_mp3": [],
        "corrupted": [],
        "duration": None
    }
    timestamp = pd.Timestamp.now()

    # Verify the files
    if n_threads > 1:
        threads = []
        chunk_size = len(df) // n_threads

        for i in range(n_threads):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_threads - 1 else len(df)

            thread = threading.Thread(target=_verify_files, args=(df.iloc[start:end], result, verbose))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
    else:
        _verify_files(df, result, verbose)

    duration = pd.Timestamp.now() - timestamp
    # formatted duration
    result["duration"] = f"{duration.seconds // 60}m {duration.seconds % 60}s"

    return result


def print_results(df, result):
    """
    Print the results of the verification.
    :param df: DataFrame with the file paths
    :param result: Result of the verification
    :return: None
    """

    if result['not_exist']:
        print(f"The following files do not exist:")
        for index in result['not_exist']:
            print(f" - {index}: {df.iloc[index]['file_path']}")
    else:
        print("All files exist.")

    if result['not_mp3']:
        print(f"The following files are not MP3 files:")
        for index in result['not_mp3']:
            print(f" - {index}: {df.iloc[index]['file_path']}")
    else:
        print("All files are MP3 files.")

    if result['corrupted']:
        print(f"The following files are corrupted:")
        for index in result['corrupted']:
            print(f" - {index}: {df.iloc[index]['file_path']}")
    else:
        print("All files are valid.")


def validate(path: pathlib.Path, verbose: bool = False):
    results = []
    files_to_delete = []

    if path.is_dir():
        files_to_verify = list(path.glob("*.csv"))

        print("Verifying files:")
        for file in files_to_verify:
            print(f" - {file}")
        print("")

        for file in files_to_verify:
            df = pd.read_csv(file)
            result = verify_data(df, verbose=verbose)

            results.append(result)
            for file_index in result["not_exist"] + result["not_mp3"] + result["corrupted"]:
                files_to_delete.append(df.iloc[file_index]["file_path"])

        print("")
        print("Results:")
        print("-" * 20)
        for file, result in zip(files_to_verify, results):
            print(f"{file} ({result['duration']})")
            print_results(pd.read_csv(file), result)
            print("")

    if files_to_delete:
        # confirm deletion
        print("Files to delete:")
        for file in files_to_delete:
            print(f" - {file}")

        print("")
        print("Do you want to delete these files? (Y/n)")
        response = input()
        if response.lower() == "n" or response.lower() == "no":
            print("Aborted.")
            exit()

        for file in files_to_delete:
            path_to_del = pathlib.Path(file)
            path_to_del.unlink()
            print(f"Deleted file from disk: {file}")

        print("Deletion finished.")
        print("")
        print("Please run 'python split.py' to remove the entries from the CSV files and balance the dataset.")
