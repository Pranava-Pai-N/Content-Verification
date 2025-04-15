import zipfile


with zipfile.ZipFile('Model_Path.zip', 'r') as zip_ref:
    zip_ref.extractall('Extracted')
    
print("Model extracted successfully!")