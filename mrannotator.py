import os
from google import genai
import pandas as pd
import re
import time
"""
FUNCTIONS
"""
def generate_text(client, system_prompt, user_prompt, temperature= 0.7):
    """
    Generates text based on given parameters via Gemini
    Input: system_prompt: str, user_prompt: str, temperature: int
    Output: response.text: str
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            config = {
                "temperature": temperature,
                "systemInstruction": system_prompt
            },
            contents= user_prompt
        )
    except Exception as e:
        print(f"Error generating for prompt {user_prompt!r}: {e!r}")
        print(f"Waiting 60s before retrying same prompt…")
        time.sleep(60)
        return generate_text(client, system_prompt, user_prompt, temperature= 0.7)

    return response.text

def iterate_through_csv(client, system_prompt, input_csv, output_csv, read_column, written_column):
    orig = pd.read_csv(input_csv)

    if os.path.exists(output_csv):
        done = pd.read_csv(output_csv).set_index("id")[written_column]
        orig[written_column] = orig["id"].map(done).fillna("")

    df = orig

    if written_column not in df.columns:
        df[written_column] = ""

    print("Total rows:", len(df), 
          "Already annotated:", (df[written_column]!="").sum(),
          "To do:", (df[written_column]=="").sum())
    
    for idx, row in df.iterrows():
        if row[written_column]:
            continue
        print("Generating for:", row[read_column])
        generated_text = generate_text(client, system_prompt, row[read_column])
        generated_text = re.sub(r"\W", "", generated_text)
        print(f"Annotated: {generated_text}")
        df.at[idx, written_column] =  "\'" + generated_text + "\'" 
        df.to_csv(output_csv, index=False)

"""
INPUT VALUES
"""
system_prompt = "You are a member of the black community. Your task is to output 'Hate' or 'Not Hate' depending the text you read"
input_csv = "sample40.csv"
output_csv = "sample40_annotated.csv"
read_column = "Input.text"
write_column = "Annotator1"

"""
MAIN CODE
"""
gemini_api_key_found = "GEMINI_API_KEY" in os.environ
print(f"Was GEMINI_API_KEY found?: {gemini_api_key_found}")

try:
    GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY")
except Exception as e:
    print("ERROR: GEMINI_API_KEY couldn't be accessed")

client = genai.Client(api_key=GEMINI_API_KEY)

iterate_through_csv(client, system_prompt, input_csv, output_csv, read_column, write_column)
