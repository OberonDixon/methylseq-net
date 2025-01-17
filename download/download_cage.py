import pandas as pd
import os
import requests
from tqdm.auto import tqdm
import argparse
from urllib.parse import urlsplit, quote, unquote
from pathlib import Path
import subprocess
import pysam
import gzip
import shutil

def download_file_with_wget(url, download_folder):
    # Create the folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Download the file using wget
    command = f"wget -P {download_folder} {url}"
    subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def sort_compress_index(filepath):
    uncompressedpath = filepath.replace('.bed.gz','.bed')
    sortedpath = filepath.replace('.bed.gz','.sorted.bed')
    sortedcompressedpath = filepath.replace('.bed.gz','.sorted.bed.gz')
    # uncompress
    with gzip.open(filepath,'rb') as input, open(uncompressedpath,'wb') as decompressed:
        shutil.copyfileobj(input,decompressed)
    # sort
    with open(sortedpath, "w") as sorted_file:
        subprocess.run(
            ["sort", "-k1,1", "-k2,2n", uncompressedpath], stdout=sorted_file
        )
    # tabix compress
    pysam.tabix_compress(sortedpath,sortedcompressedpath,force=True)
    # tabix index
    pysam.tabix_index(sortedcompressedpath, preset="bed",force=True)

parser = argparse.ArgumentParser(description='Download, format, and index CAGE file based on which ones are needed.')
parser.add_argument('--save_directory', type=str, required=True, help='Directory to save the files.')
args = parser.parse_args()

download_folder = args.save_directory

primary_cell_hCAGE = pd.read_csv('00_human.primary_cell.hCAGE.hg38.assay_sdrf.tsv',sep='\t',header=0)
methyl_to_atlas = pd.read_csv('../configs/cell_type_matching_atac-cage.tsv',sep='\t',header=0)

url_prefix = 'https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_v9/basic/human.primary_cell.hCAGE/'

download_urls = []
for CAGE_library_list in methyl_to_atlas['CAGE_Atlas_Library']:
    try:
        for CAGE_library in CAGE_library_list.split(','):
            download_urls.append(url_prefix + quote(primary_cell_hCAGE[primary_cell_hCAGE['Library Name']==CAGE_library]['File Name.1'].iloc[0].replace('.bam','.ctss.bed.gz')))
    except:
        continue

print(download_urls)

for download_url in tqdm(download_urls):
    filename = unquote(unquote(os.path.basename(urlsplit(download_url).path)))
    filepath = Path(download_folder) / filename
    index_filename = filename.replace('.bed.gz','.sorted.bed.gz.tbi')
    index_filepath = Path(download_folder) / index_filename
    # print(download_url)
    if os.path.exists(index_filepath):
        print(filename,"already downloaded and indexed.")
    else:
        download_file_with_wget(download_url,download_folder)
        sort_compress_index(str(filepath))