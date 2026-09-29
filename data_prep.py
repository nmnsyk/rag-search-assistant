import pandas as pd
import json
import re
import requests
from io import BytesIO
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions



experiments = pd.read_json("raw_data.json")
categories = pd.read_json("categories.json")


# Filter experiments that start with "P" and create experiments DataFrame
just_experiments = experiments[experiments["experiment"].str.startswith("P", na=False)].copy()


# Deu_translations with desc_2
deu_translations = []

for translations_list in just_experiments["translations"]:
    found_desc = "" 
    
    if isinstance(translations_list, list):
        for translation in translations_list:
            if translation.get("language_code") == "DEU":
                found_desc = translation.get("desc_2", "")
                break 
                
    deu_translations.append(found_desc) 
    
just_experiments["desc_2"] = deu_translations      



# Metadata filter sub_characteristics - values 
wanted_characteristics = [
    "Vorbereitungszeit (in Minuten)", 
    "Durchführungszeit (in Minuten)",
    "Gruppengröße", 
    "Artikelniveau", 
    "Digital/analog", 
    "Schwierigkeitsgrad"
]

all_extracted_characteristics = []

for characteristics_list in just_experiments["characteristics"]: 
   characteristics_dict = {}
    
   if type(characteristics_list) == list:
      for characteristic in characteristics_list:
         name = characteristic.get("sub_characteristic")
         
         if name in wanted_characteristics:
            values = []
            
            value_1 = str(characteristic.get("value_1", "")).strip()
            if value_1 != "": values.append(value_1)

            value_2 = str(characteristic.get("value_2","")).strip()
            if value_2 != "": values.append(value_2)

            value_3 = str(characteristic.get("value_3", "")).strip()
            if value_3 != "": values.append(value_3)
            
            characteristics_dict[name] = " , ".join(values)
            
   all_extracted_characteristics.append(characteristics_dict)

just_experiments["extracted_characteristics"] = all_extracted_characteristics




# CATEGORY MAPPING ->   AKEBIO : Biologie...

category_dictionary = {}
index = 0

for code in categories["Category_Code"]:
   # Get the category name for this row 
   name = categories["Category"][index]
   # Save it to the dictionary {'AKPH' : 'Webshop'}
   category_dictionary[code] = name
   index = index + 1

all_mapped_categories = []

for row_data in just_experiments["categories"]:
   row_results = []
   
   if type(row_data) == list:
      for path in row_data:
         # split the string into single codes = ["AKPH;AKEBIO"] -> ["AKPH", "AKEBIO"]
         codes_list = path.split(";")
         names_list = []
         
         # Check each code and find its name in the dictionary
         for code in codes_list:
            if code in category_dictionary:
               name = category_dictionary[code]

               if not re.search(r"P\d+", name) and name != "Webshop": 
                  names_list.append(name)
                 

         # Join the names with ' > ' ("Webshop > Physik")
         text_with_arrows = " > ".join(names_list)
         row_results.append(text_with_arrows)

   # If there are multiple paths, join them with " | "
   final_text = " | ".join(row_results)
   # Add the final string to our main list
   all_mapped_categories.append(final_text)
just_experiments["categories_mapped"] = all_mapped_categories


# Read extracted_texts and create a new column 
pdf_texte = pd.read_json("pdf_texte.json")
just_experiments["extracted_text"] = pdf_texte.iloc[:, 0].values



# Extracted Text Filtration with REGEX patterns
patterns = [
    r"P\d{7}",                                         
    r"\d{5}\s...",                               
    r"Fax:\s*[\d\s\-]+",                               
    r"This can also be found online at:\s*",        
    r"This can also be at:\s*",                        
    r"This content can also be found ... at:\s*",   
    r"http[s]?:\\?\/\\?\/[^\s]+|http[s]?://[^\s]+",    
    r"\b\d{1,2}\\?/\d{1,2}\b",                         
    r"[☑☐O]",                                  
    r"Schwierigkeitsgrad.*?Durchführungszeit",         
    r"\d+\+?\s*Minuten",                               
    r"Material\s*Position.*?Menge.*?(?=Aufbau|Durchführung)", 
    r"Protokoll.*",                                   
    r"(?:leicht|mittel|schwer)\s*\d+",               
    r"\(\s*\)",                                     
    r"\(z\.B\.\s*nach\s*\)",                           
]

cleaned_text = []
for text in just_experiments["extracted_text"]:
    if isinstance(text, str):
        new_text = text.replace("\n", " ")  
   
        for p in patterns:
            if re.search(p, new_text):
                new_text = re.sub(p, "", new_text)
                
        cleaned_text.append(new_text)
    else:
        cleaned_text.append(text)
        
just_experiments["extracted_text"] = cleaned_text


# create a final DataFrame
final_df = just_experiments[["experiment","title", "desc_2"]].copy()
#rename desc_2
final_df = final_df.rename(columns={"desc_2": "subtitle"})
final_df["characteristics"] = just_experiments["extracted_characteristics"]
final_df["categories"] = just_experiments["categories_mapped"]
final_df["extracted_text"] = just_experiments["extracted_text"]
final_df.info()
# convert the dataframe into the requested JSON structure
final_df.to_json("final_cleaned_data.json", orient="records", force_ascii=False, indent=4)


# ChromaDB Insertion
client = chromadb.PersistentClient(path="./phywe_vectordb")

embedding_fc = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection = client.get_or_create_collection(
    name="phywe_db",
    embedding_function=embedding_fc,
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)

# chromadb needs documents (title, extracted_text), metadatas (subtitle, characteristics, categories) and ids (P numbers) as lists
documents = []
metadatas = []
ids = []

for index, row in final_df.iterrows():
    # title
    if pd.notna(row["title"]): 
        title = str(row["title"])
    else:
        title = ""

    # extracted_text
    if pd.notna(row["extracted_text"]):
        extracted_text = str(row["extracted_text"])
    else:
        extracted_text = ""

#  document text: title + categories + extracted_text        
    doc_result = "Title: " + title  + " Content: " + extracted_text
    documents.append(doc_result)


# METADATA: subtitle + characteristics + categories
    metadata = {}

    if pd.notna(row["subtitle"]):
        metadata["subtitle"] = str(row["subtitle"])
    else:
        metadata["subtitle"] = ""

# categories mapping to list (Biologie > Webshop) -> ["Biologie", "Webshop"]
    all_categories = []

    if pd.notna(row["categories"]):
        cleaned_text = str(row["categories"]).replace(" | ", " > ")
        parts = cleaned_text.split(" > ")
        
        for part in parts:
            clean_part = part.strip()
            
            if clean_part != "":
                if clean_part not in all_categories:
                    all_categories.append(clean_part)

    if len(all_categories) > 0:
        metadata["categories"] = all_categories

# characteristics 
    characteristics_data = row["characteristics"]
    if isinstance(characteristics_data, dict):
        for key, value in characteristics_data.items():
            metadata[key] = value
    metadatas.append(metadata)

# P number as id
    if pd.notna(row["experiment"]):
        ids.append(str(row["experiment"]))
    else:
        ids.append(str(index))

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)   

print("Data inserted into ChromaDB successfully!")