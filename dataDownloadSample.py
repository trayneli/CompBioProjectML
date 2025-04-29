import requests
import pandas as pd
import os

output=pd.read_table('Annotation_EC_Data.tsv')
#outputCurrent=pd.read_table('currentFiles', header=None)

# Create directory if it doesn't exist
if not os.path.exists('ProteinData'):
    os.makedirs('ProteinData')

## Download based on uniprotID
for i in output['UniProtID']:
    url = f"https://rest.uniprot.org/uniprotkb/{i}.fasta"
    response = requests.get(url)
    if response.ok:
        with open('ProteinData/' + f"{i}.fasta", "w") as f:
            f.write(response.text)
        print(f"Downloaded {i} sequence")
    else:
        print(f"Failed to download: {response.status_code}")
        