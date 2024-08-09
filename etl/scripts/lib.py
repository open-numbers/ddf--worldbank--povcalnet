import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def download(url, filename, num_retries=3):
    # Set up a retry strategy in case of connection issues
    retry_strategy = Retry(
        total=num_retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=3
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)


    file_to_download = open(filename, 'wb')
    with http.get(url, stream=True) as response:
        response.raise_for_status()  # Check for HTTP issues
        for chunk in response.iter_content(chunk_size=8192):
            file_to_download.write(chunk)

    file_to_download.close()
    print(f"Downloaded {filename} successfully.")
