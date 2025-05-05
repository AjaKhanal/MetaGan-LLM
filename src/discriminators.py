import json

from ollama import ChatResponse, chat
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

import json
import pandas as pd


from stats_library import StatsLibrary


def generate_discriminator_prompt(real_data, synthetic_data):

    try:
        real_df = process_data(real_data, 'real_data')
        synthetic_df = process_data(synthetic_data, 'synthetic_data')

        stats = StatsLibrary(real_df, synthetic_df)
        comparison = stats.prompt_formation()
        print(comparison)
    except Exception as e:
        print(e)
        comparison = ""



    sample_output = '{"Type": "Real/Synthetic", "Feedback": "Single sentence to make the data more realistic"}'
    return f"""
    Guaranteed Real Data Sample:
    {real_data}

    Undetermined Data Sample:
    {synthetic_data}
    
    Statistics:
    {comparison}
    
    Is the undetermined data sample real or synthetic? Provide a single word response ONLY: Real/Synthetic.
    If it is synthetic give your feedback to make the data more realistic in a single sentence, else leave it blank. 
    Answer in the following JSON format {sample_output}
    """


def statistical_gpt(real_data, synthetic_data, _model="gpt-4o-mini"):
    load_dotenv(find_dotenv())
    client = OpenAI(
        api_key=os.environ.get('OPENAI_API_KEY'),
    )

    discriminator_prompt = generate_discriminator_prompt(real_data, synthetic_data)

    completion = client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": discriminator_prompt}
        ]
    )

    ans = completion.choices[0].message.content
    return ans


def statistical_llama(real_data, synthetic_data, _model="llama3.1"):
    discriminator_prompt = generate_discriminator_prompt(real_data, synthetic_data)
    response: ChatResponse = chat(
        model=_model,
        messages=[
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": discriminator_prompt}
        ]
    )

    ans = response.message.content
    return ans


def explicit(real_data, synthetic_data, _model="llama3.1"):
    print("hi")
    # todo: code it up
    discriminator_prompt = generate_discriminator_prompt(real_data, synthetic_data)
    response: ChatResponse = chat(
        model=_model,
        messages=[
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": discriminator_prompt}
        ]
    )

    ans = response.message.content
    return ans


def adversarial(real_data, synthetic_data):
    print("hi")
    # todo: code it up



def process_data(data, filename):
    os.makedirs('data', exist_ok=True)
    try:
        data_json = json.loads(data)
        json_filename = f"data/{filename}.json"
        with open(json_filename, 'w') as json_file:
            json.dump(data_json, json_file)
        return pd.read_json(json_filename)
    except json.JSONDecodeError:
        csv_filename = f"data/{filename}.csv"
        with open(csv_filename, 'w') as csv_file:
            csv_file.write(data)
        return pd.read_csv(csv_filename)

