import chromadb
import json
import ollama
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi  

with open("final_cleaned_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# BM25 Search Function
tokenized_corpus = []
for item in data:
    title = str(item.get("title", "")).lower()
    subtitle = str(item.get("subtitle", "")).lower()
    text = f"{title} {subtitle}"
    tokenized_corpus.append(text.split())
        
bm25 = BM25Okapi(tokenized_corpus)
    
def bm25_search(user_query):    
    query_tokens = user_query.lower().split()
    scores = bm25.get_scores(query_tokens)
    bm25_results = []
    for i, score in enumerate(scores):
        if score > 0:
            bm25_results.append({
                "experiment": data[i].get("experiment"),
                "title": data[i].get("title"),
                "subtitle": data[i].get("subtitle"),
                "metadata": data[i].get("characteristics", {}),
                "categories": data[i].get("categories",""),
                "score": round(score, 4) 
            })

    bm25_results.sort(key=lambda x: x["score"], reverse=True)
    return bm25_results     

# ChromaDB Persistent Client and Collection Setup
client = chromadb.PersistentClient(path="./phywe_vectordbbge")
embedding_fc = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
collection = client.get_collection(name="phywe_dbbge", embedding_function=embedding_fc)

user_query = "Gib mir Experimente zum Thema Wärmeleitung, die für Schüler der Klasse 7-10 geeignet sind und eine Durchführungszeit von 30 Minuten haben."

prompt = """Read the user query and answer only with valid JSON. No prose, no markdown.

Return exactly this schema:
{
    "semantic_search": "Extract all relevant search keywords from the user query. Include important objects, materials, actions, physical or chemical concepts, and phenomena. Keep specific technical terms (e.g. 'Krokodilklemmen', 'Butanbrenner', 'Eiswürfel'). Do not shorten too aggressively. Aim for 4–8 meaningful keywords. Do NOT include metadata-only constraints such as exact duration, group size, difficulty, or article level. Return semantic_search always as a single string. Never return a list.",
    "metadata": {
    "subtitle": null,
    "categories": null,
    "Schwierigkeitsgrad": null,
    "Durchführungszeit (in Minuten)": null,
    "Digital/analog": null,
    "Gruppengröße": null,
    "Artikelniveau": null
  }
}

CRITICAL RULES:
1. Do NOT guess or infer metadata.
2. If the user does not explicitly mention a metadata value, use null.
3. Never infer categories from topic words.
4. Only set categories if the user explicitly says: Physik, Chemie, Biologie. Return ONLY the string value for categories (e.g., "Physik").
5. Only set Schwierigkeitsgrad if the user explicitly says: leicht, mittel, schwer
6. Only set Durchführungszeit (in Minuten) if the user explicitly gives a number of minutes.
7. Only set Gruppengröße if the user explicitly gives a number of people or says "zu zweit" (= "2"). Return ONLY the string value for Gruppengröße (e.g.,"2"). 
8. Only set Digital/analog if the user explicitly says "digital" or "analog".
9. Only set Artikelniveau if the user explicitly mentions specific class grades or university. The output MUST be exactly one of these predefined values: "Klasse 5-7", "Klasse 7-10", "Klasse 10-13", "Hochschule". Map the user's input to the correct category (e.g., if user says "Klasse 8", output "Klasse 7-10").
10. Do NOT infer Artikelniveau from words like: Schüler, Kinder, Schule.
11. If unsure, use null.
"""

response = ollama.chat (
    model = "llama3.1",
    messages= [ {"role": "system", "content": prompt},
                {"role": "user", "content": user_query}
              ],
    format = "json"           
)

print("User Query:")
print(user_query)
try:
    json_response = json.loads(response['message']['content'])
    print("JSON Response from LLM:")
    print(json.dumps(json_response, indent=4, ensure_ascii=False))
except:
    json_response = {} 
    print("LLM can not provide a valid JSON response")


search_text = json_response.get("semantic_search", user_query)
filters = json_response.get("metadata", {})


lexical_results = bm25_search(user_query)
print("=======================================================")
print("\nBM25 Lexical Search Results (Title + Subtitle): ")
print("\n")
for item in lexical_results[:5]: 
    print(f"Experiment-ID: {item.get('experiment')}")
    print(f"Title: {item.get('title')}")
    print(f"Subtitle: {item.get('subtitle')}")
    print(f"Score: {item.get('score')}")
    print("\n")


#if bedingung einfügen
clean_filters = []
for key, value in filters.items():
    if value != "" and value is not None:
        if isinstance(value, list):
            if len(value) > 0:
                clean_filters.append((key, str(value[0]))) 
        else:
            clean_filters.append((key, str(value))) 

    
and_conditions = []
for key, value in clean_filters:
    if isinstance(value, list):
        if len(value) == 0:
            continue 
        value = str(value[0])

    if key == "categories": 
        and_conditions.append({key: {"$contains": value}})
    else:
        and_conditions.append({key: value})
        
if len(and_conditions) == 0:
    where_condition = None
elif len(and_conditions) == 1:
    where_condition = and_conditions[0]
else:
    where_condition = {"$and": and_conditions}

results = collection.query(
    query_texts=[search_text],
    n_results=5,
    where=where_condition,
    include=["documents", "metadatas", "distances"]
)

print("=======================================================")
print("Semantic search Results (ChromaDB):")
if len(results["documents"][0]) > 0:
    for i in range(len(results["documents"][0])):
        print(f"Result {i+1}:")
        print(f"Document: {results['documents'][0][i]}")
        print(f"Metadata: {results['metadatas'][0][i]}")
        print(f"Distance: {results['distances'][0][i]}")
        print("\n")
else:
    print("No semantic results found with the given filters.")




# Answer generation with LLM based on retrieved documents
print("Generating answer with LLM")
print(f"User Query: {user_query}")

#keyword search results
keyword_context = ""
for item in lexical_results[:5]:
    keyword_context += f"Experiment-ID: {item['experiment']}\n"
    keyword_context += f"Titel: {item['title']}\n"
    keyword_context += f"Untertitel: {item['subtitle']}\n"
    keyword_context += f"Keyword-Score: {item['score']}\n"
    keyword_context += "---------------------------------\n"

#semantic search results 
context_text = ""
found_ids = results["ids"][0]
found_docs = results["documents"][0]
found_metas = results["metadatas"][0]
found_distances = results["distances"][0]

if len(found_docs) == 0:
        context_text = "Keine passenden Datensätze gefunden."
else:
    for doc_id, doc_text, meta, dist in zip(found_ids, found_docs, found_metas, found_distances):
       
        clean_doc = doc_text[:800]
        clean_doc = clean_doc.replace("Title:", "Titel:").replace("Content:", "\nInhalt:")

        context_text += f"Experiment-ID: {doc_id}\n"
        context_text += f"Content: {clean_doc}\n"
        context_text += f"Metadaten: {meta}\n"
        context_text += f"Distance: {dist}\n"

        context_text += "---------------------------------------------\n"

answer_prompt = """
You are an internal assistant for an experiment database.

Answer the user's query using only the provided search results.
Answer in German.

First decide the intent:

1. If the user is looking for an experiment, recommend the best matching experiments.
Use this format:
Hier sind die passendsten Experimente:

1. [Titel]
- Experiment-ID: ...

2. If the user asks for specific information from an experiment, answer the question directly.
Do not recommend experiments in this case.
Use this format:
Für den Versuch "[Titel]" ([Experiment-ID]):

[Direkte Antwort]

Rules:
- Use only the provided search results.
- Do not invent titles, IDs, values, settings, materials, or explanations.
- Ignore irrelevant results.
- Prefer results that match the main topic of the query.
- If the answer is not contained in the results, say:
"Diese Information konnte ich in den Suchergebnissen nicht sicher finden."
- Keep the answer short and useful.
"""

user_prompt = f"""    

USER QUERY:
{user_query}

SEMANTIC RESULTS: 
{context_text}

KEYWORD RESULTS:
{keyword_context}

- Prefer exact or very close title/subtitle matches from keyword_results when clearly relevant.
- Ignore irrelevant results completely.
- Do not mention all retrieved experiments.
"""

answer_response = ollama.chat(
        model="llama3.1",
        messages=[
        {"role": "system", "content": answer_prompt},
        {"role": "user", "content": user_prompt}
    ])

print("========================================================================================")
print(answer_response['message']['content'])

