import re
import subprocess
import openai
import numpy as np

def get_table_info(table_fname):
    script = f"import pandas as pd\ndf = pd.read_csv('{table_fname}')\n" \
             f"print('- Path to the table: `{table_fname}`')\n" \
              "print('\\n- First five rows:')\nprint(df.head(5).to_string())\n" \
              "print('\\n- Column types:')\nprint(df.dtypes)"
    info_text, error = python(script)
    assert info_text and not error
    return info_text

def get_user_prompt(text):
    return {"role": "user", "content": text}

def get_assist_prompt(text):
    return {"role": "assistant", "content": text}

def dsGPT(prompts, openai_model):
    print('\n' + '-' * 115)

    print("\n********** Prompts for dsGPT **********")
    for prompt in prompts:
        print(f"\n[{prompt['role'].upper()}]\n")
        print(prompt["content"])
    
    openai_response = openai.ChatCompletion.create(
        model=openai_model,
        messages=[{"role": "system", "content": "You are now a data scientist."}, *prompts]
    )
    token_count = openai_response.usage.total_tokens
    
    print("\n**********  Response from dsGPT **********")
    text = "\n\n".join([c.message.content for c in openai_response.choices])
    print(f"\n[dsGPT]\n")
    print(text)
    print('\n' + '-' * 115)
    
    return text, token_count

def devGPT(prompts, openai_model):
    print('\n' + '-' * 115)
    
    print("\n********** Prompts for devGPT **********")
    for prompt in prompts:
        print(f"\n[{prompt['role'].upper()}]\n")
        print(prompt["content"])
    
    openai_response = openai.ChatCompletion.create(
        model=openai_model,
        messages=[{"role": "system", 
                   "content": "You are now a Python developer. You MUST avoid calling any graph outputting function as the environment lacks a GUI."}, 
                  *prompts]
    )
    token_count = openai_response.usage.total_tokens
    
    print("\n**********  Response from devGPT **********")
    text = "\n\n".join([c.message.content for c in openai_response.choices])
    print(f"\n[devGPT]\n")
    print(text)
    print('\n' + '-' * 115)
    
    return text, token_count

def editGPT(prompts, openai_model):
    print('-' * 115)
    
    print("\n********** Prompts for editGPT **********")
    for prompt in prompts:
        print(f"\n[{prompt['role'].upper()}]\n")
        print(prompt["content"])
    
    openai_response = openai.ChatCompletion.create(
        model=openai_model,
        messages=[{"role": "system", "content": "You are now an editor with knowledge of data science."}, *prompts]
    )
    token_count = openai_response.usage.total_tokens
    
    print("\n**********  Response from editGPT **********")
    text = "\n\n".join([c.message.content for c in openai_response.choices])
    print(f"\n[editGPT]\n")
    print(text)
    print('\n' + '-' * 115)
    
    return text, token_count

def extract_code(text):
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    code = match.group(1).strip().strip("python\n").strip("Python\n") if match else None
    return code

def python(script):
    print('-' * 115)
    
    result = subprocess.run(["python3", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = result.stdout.decode("utf-8")
    err_str = result.stderr.decode("utf-8")
    
    print(f"\n********** {'Error' if len(err_str) else 'Output'} from Python **********")
    print(out_str)
    print(err_str)
    print('\n' + '-' * 115)
    return out_str, err_str
