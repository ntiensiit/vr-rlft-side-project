# Copyright 2015 Yale University - Grablab
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Modified to work with Python 3 by Sebastian Castro, 2020

import json
import os
from urllib.request import Request, urlopen

# Define an output folder
output_directory = os.path.join("data", "raw", "ycb")

# Define a list of objects to download from
# http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/
objects_to_download = [
    "003_cracker_box",
    "004_sugar_box",
    "006_mustard_bottle",
]

# You can edit this list to only download certain kinds of files.
# 'berkeley_rgbd' contains all of the depth maps and images from the Carmines.
# 'berkeley_rgb_highres' contains all of the high-res images from the Canon cameras.
# 'berkeley_processed' contains all of the segmented point clouds and textured meshes.
# 'google_16k' contains google meshes with 16k vertices.
# 'google_64k' contains google meshes with 64k vertices.
# 'google_512k' contains google meshes with 512k vertices.
# See the website for more details.
# files_to_download = [
#     "berkeley_rgbd",
#     "berkeley_rgb_highres",
#     "berkeley_processed",
#     "google_16k",
#     "google_64k",
#     "google_512k",
# ]
files_to_download = ["berkeley_processed", "google_16k"]

# Extract all files from the downloaded .tgz, and remove .tgz files.
# If false, will just download all .tgz files to output_directory
extract = True

base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
objects_url = "https://ycb-benchmarks.s3.amazonaws.com/data/objects.json"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def fetch_objects(url):
    """ Fetches the object information before download """
    response = urlopen(url)
    html = response.read()
    objects = json.loads(html)
    return objects["objects"]


def download_file(url, filename, max_retries=5):
    """ Downloads files from a given URL with retries """
    import time
    for attempt in range(max_retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=30) as u:
                file_size_header = u.getheader("Content-Length")
                file_size = int(file_size_header) if file_size_header else 0
                print(f"Downloading: {filename} ({file_size / 1000000.0} MB)")

                with open(filename, "wb") as f:
                    file_size_dl = 0
                    block_sz = 65536
                    while True:
                        buffer = u.read(block_sz)
                        if not buffer:
                            break

                        file_size_dl += len(buffer)
                        f.write(buffer)
                        if file_size > 0:
                            status = (
                                f"{int(file_size_dl / 1000000.0):10d}  "
                                f"[{file_size_dl * 100.0 / file_size:3.2f}%]"
                            )
                        else:
                            status = f"{int(file_size_dl / 1000000.0):10d}"
                        status = status + chr(8) * (len(status) + 1)
                        print(status)
        except Exception as e:
            print(f"Error downloading (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise
        else:
            return

def tgz_url(object, type):
    """ Get the TGZ file URL for a particular object and dataset type """
    if type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return f"{base_url}berkeley/{object}/{object}_{type}.tgz"
    if type == "berkeley_processed":
        return f"{base_url}berkeley/{object}/{object}_berkeley_meshes.tgz"
    return f"{base_url}google/{object}_{type}.tgz"


def extract_tgz(filename, dir):
    """ Extract a TGZ file using built-in tarfile """
    import tarfile
    import time
    with tarfile.open(filename, "r:gz") as tar:
        if hasattr(tarfile, 'data_filter'):
            tar.extractall(path=dir, filter='data')
        else:
            tar.extractall(path=dir)

    # Retry removing the file in case of Windows antivirus locks (WinError 32)
    for _ in range(10):
        try:
            os.remove(filename)
            break
        except OSError:
            time.sleep(1)


def check_url(url):
    """ Check the validity of a URL """
    try:
        request = Request(url)
        request.get_method = lambda: "HEAD"
        with urlopen(request):
            pass
    except Exception:
        return False
    else:
        return True


if __name__ == "__main__":
    import shutil

    # Grab all the object information
    objects = fetch_objects(objects_url)

    # Download each object for all objects and types specified
    for object in objects:
        if objects_to_download == "all" or object in objects_to_download:
            object_dir = os.path.join(output_directory, object)

            try:
                for file_type in files_to_download:
                    # Check if already downloaded/extracted
                    if extract:
                        is_google_extracted = (
                            file_type == "google_16k"
                            and os.path.exists(os.path.join(object_dir, "google_16k"))
                        )
                        is_berkeley_extracted = (
                            file_type == "berkeley_processed"
                            and os.path.exists(os.path.join(object_dir, "clouds"))
                        )
                        if is_google_extracted or is_berkeley_extracted:
                            print(f"Skipping {object} {file_type}: already extracted.")
                            continue
                    else:
                        filename = f"{output_directory}/{object}_{file_type}.tgz"
                        if os.path.exists(filename):
                            print(f"Skipping {object} {file_type}: already downloaded.")
                            continue

                    url = tgz_url(object, file_type)
                    if not check_url(url):
                        continue

                    filename = f"{output_directory}/{object}_{file_type}.tgz"

                    download_file(url, filename)
                    if extract:
                        extract_tgz(filename, output_directory)

            except Exception as e:
                import glob
                import time

                print(f"Failed to process {object}: {e}. Cleaning up...")
                if os.path.exists(object_dir):
                    shutil.rmtree(object_dir, ignore_errors=True)

                # Remove any leftover .tgz files related to this object
                tgz_pattern = os.path.join(output_directory, object + "_*.tgz")
                for f in glob.glob(tgz_pattern):
                    for _ in range(5):
                        try:
                            os.remove(f)
                            break
                        except OSError:
                            time.sleep(1)
                continue