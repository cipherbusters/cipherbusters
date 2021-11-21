import gdown
import shutil
import os

url = 'https://drive.google.com/file/d/1rO66UfZ0H8QPvqcKX8Enkp51SAYBxShb/view?usp=sharing'
output = 'data.zip'
gdown.download(url, output, quiet=False, fuzzy=True)
print('Extracting zip file...')
shutil.unpack_archive(output, '.')
print('Deleting archive...')
os.remove(output)
