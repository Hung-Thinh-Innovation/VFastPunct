from vfastpunct.constants import LOGGER

import requests


def download_file_from_google_drive(id, destination, confirm=None):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    LOGGER.info(f"Download pretrained model..")
    URL = "https://docs.google.com/uc?export=download"
    if confirm is not None:
        URL += f"&confirm={confirm}"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


# Debug
if __name__ == "__main__":
    download_file_from_google_drive('1Iv3iQfuA7NWRa2lQgWVn4fMLk4-XkWwZ', './mbertpunccap.pt', 't')