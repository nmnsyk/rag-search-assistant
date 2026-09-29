import pandas as pd
import json
import requests
from io import BytesIO
from pypdf import PdfReader


experiments = pd.read_json("raw_data.json")
categories = pd.read_json("categories.json")


# Filter experiments that start with "P" and create experiments DataFrame
just_experiments = experiments[experiments["experiment"].str.startswith("P", na=False)].copy()


#PDF EXTRACTION 

extracted_manual_urls = []

for files_list in just_experiments["files"]:
    found_url = ""
    
    if type(files_list) == list:
        for file_data in files_list:
            
            file_url = file_data.get("file_url", "")
            if 'phy_itemtestinstruction' in file_url and '_de' in file_url:
                    found_url = file_url
                    break              
    extracted_manual_urls.append(found_url)

just_experiments["clean_manual_url"] = extracted_manual_urls


def extract_pdf_text(pdf_url):
    try:
        response = requests.get(pdf_url, timeout=10)
        
        if response.status_code != 200:
            return ""
            
        reader = PdfReader(BytesIO(response.content))
        extracted_text = ""
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text != None:  
                extracted_text = extracted_text + page_text + " "
                
        return extracted_text.strip()
        
    except:
        return ""  


# Extract text from ALL URLs

final_texts = []

for url in just_experiments["clean_manual_url"]:
    if type(url) == str and url != "":
        text = extract_pdf_text(url)
        final_texts.append(text)
    else:
        final_texts.append("")

just_experiments["extracted_text"] = final_texts


# Export to JSON
save_file_name = "pdf_texte.json"

just_experiments["extracted_text"].to_json(save_file_name, orient="records", force_ascii=False, indent=4)

