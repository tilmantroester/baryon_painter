import io
import os

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

def get_folder_contents(service, folder_id):
    """From https://stackoverflow.com/questions/36874223/why-wont-google-api-v3-return-children"""
    query = f"'{folder_id}' in parents"

    response = service.files().list(q=query,
                                    spaces='drive',
                                    fields='files(id, name, size)').execute()

    return response["files"]

def download_file(service, file_id, file_name, file_size, download_path):
    print(f"Downloading {file_name} ({int(file_size)/1024**2:.1f} MB):", end="")

    request = service.files().get_media(fileId=file_id)
    downloaded = io.BytesIO()
    downloader = MediaIoBaseDownload(downloaded, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        if status:
            progress_percentage = status.progress() * 100
            print(f" {progress_percentage:.1f}%", end="")
    print("")

    downloaded.seek(0)
    
    output_fn = os.path.join(download_path, file_name)
    with open(output_fn, "wb") as f:
        f.write(downloaded.read())

def download_files_in_folder(folder_id, download_path, exclude_filters=[]):
    """Download all files in a folder.
    
    Arguments
    ---------
    folder_id : str
        Google Drive ID for the folder whose content should be downloaded.
    download_path : str
        Path where the files should be written to.
    exclude_filters : list, optional
        List of strings that if found as a substring in a file name cause that file to be excluded from the download.
    """
    drive_service = build('drive', 'v3')

    file_infos = get_folder_contents(drive_service, folder_id)
    # Sort by filename
    file_infos = sorted(file_infos, key=lambda k: k["name"])
    
    print(f"Writing downloaded files to {download_path}")
    for file_info in file_infos:
        for f in exclude_filters:
            if f in file_info["name"]:
                break
        download_file(drive_service, 
                      file_info["id"], file_info["name"], file_info["size"], 
                      download_path)
    print("Done!")