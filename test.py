import argparse
import sys
import os.path as osp
import pandas as pd
import openai

from utils import *

openai.api_key = open("api_key.txt", 'r').read().strip()
OPENAI_MODEL = "gpt-3.5-turbo"
TOKEN_COUNT = 0

def anomalyDetection(table_fname):
    global TOKEN_COUNT
    text = f"Here is some information of a table.\n" \
           f"Table Information:\n\"\"\"\n{get_table_info(table_fname)}\n\"\"\""
    prompt_data = get_user_prompt(text)
    ques = "Where are significant anomalies in the table?\n" \
           "Typically, an anomaly occurs when there is a sudden spike or drop in a numeric column."
    text = f"Here's a question:\n\"\"\"\n{ques}\n\"\"\""
    prompt_ques = get_user_prompt(text)
    
    text = "Please write a Python script helping answer the question.\n" \
           "In the script, please print ALL the following materials:\n" \
           "- Enriched results of analyses\n" \
           "- Reasons why the results can help answer the question"
    prompt_code = get_user_prompt(text)
    output, error = '', '.'
    while error or not output:
        response_code, count = devGPT([prompt_data, prompt_ques, prompt_code], OPENAI_MODEL)
        TOKEN_COUNT += count
        code = extract_code(response_code)
        if not code:
            continue
        output, error = python(code)

    text = "Here are the Python code and its output that help answer the question.\n\n" \
          f"Code:\n\"\"\"\n{code}\n\"\"\"\n\n" \
          f"Output:\n\"\"\"\n{output}\n\"\"\""
    prompt_output = get_user_prompt(text)
    text = "Please answer the question with a numbered list of anomaly descriptions."
    prompt_ans = get_user_prompt(text)
    response_ans, count = dsGPT([prompt_ques, prompt_output, prompt_ans], OPENAI_MODEL)
    TOKEN_COUNT += count

    return response_ans

def causeDiscovery(issue, target_fname, supp_fname, num_qas):
    global TOKEN_COUNT
    text = f"Here is some information of the Target table.\n\n" \
           f"Target Table Information:\n\"\"\"\n{get_table_info(target_fname)}\n\"\"\""
    prompt_target = get_user_prompt(text)
    text = f"Here is some information of the Support table, which is a breakdown of the Target table.\n\n" \
           f"Support Table Information:\n\"\"\"\n{get_table_info(supp_fname)}\n\"\"\""
    prompt_supp = get_user_prompt(text)
    
    text = f"Here's an issue discovered in the Target table:\n\"\"\"\n{issue}\n\"\"\""
    prompt_issue = get_user_prompt(text)
    text = "Task: To find root causes of the issue in the Target table through analyses of the information in the Support table"
    prompt_goal = get_user_prompt(text)
    
    qas = []
    while len(qas) < num_qas:
        if len(qas) == 0:
            text = "Please propose one question that would aid in accomplishing the task.\n" \
                   "Note that the question MUST be answerable using the given tables and Python."
            prompt_ques = get_user_prompt(text)
            text, count = dsGPT([prompt_target, prompt_supp, prompt_issue, prompt_goal, prompt_ques], OPENAI_MODEL)
            TOKEN_COUNT += count
            response_ques = get_assist_prompt(text)
        else:
            qas_str = "\n\n".join([f"Question {i}:\n\"\"\"\n{qa['Q']}\n\"\"\"\n" \
                                   f"Answer {i}:\n\"\"\"\n{qa['A']}\n\"\"\"" for i, qa in enumerate(qas)])
            text = f"Here are existing questions and their corresponding answers for the task:\n\n{qas_str}"
            prompt_his = get_user_prompt(text)
            text = "Is there any follow-up question that would aid in accomplishing the task?\n" \
                   "IF YES, please directly provide one.\n" \
                   "OTHERWISE, please simply answer with the word \"NO\".\n" \
                   "Note that the question MUST be answerable using the given tables and Python."
            prompt_ques = get_user_prompt(text)
            text, count = dsGPT([prompt_target, prompt_supp, prompt_issue, prompt_goal, prompt_his, prompt_ques], OPENAI_MODEL)
            TOKEN_COUNT += count
            if text.strip()[:2].lower() == "no":
                break
            response_ques = get_assist_prompt(text)
        
        output, error = '', '.'
        while error or not output:
            text = "Please write a Python script helping answer the question.\n" \
                   "Please print enriched results of analyses " \
                   "and provide short reasons why the results can help answer the question."
            prompt_code = get_user_prompt(text)
            text, count = devGPT([prompt_target, prompt_supp, prompt_issue, prompt_goal, prompt_ques, 
                                  response_ques, prompt_code], OPENAI_MODEL)
            TOKEN_COUNT += count
            response_code = get_assist_prompt(text)
            code = extract_code(text)
            if not code:
                continue
            output, error = python(code)
        
        text = f"Here is the actual output of the code:\n\"\"\"\n{output}\n\"\"\""
        response_output = get_assist_prompt(text)
        text = "Please provide an answer to the question for achieving the task."
        prompt_ans = get_user_prompt(text)
        text, count = dsGPT([prompt_target, prompt_supp, prompt_issue, prompt_goal, 
                             prompt_ques, response_ques, prompt_code, response_code, 
                             response_output, prompt_ans], OPENAI_MODEL)
        TOKEN_COUNT += count
        qas.append({'Q': response_ques["content"], 'A': text})

    qas_str = "\n\n".join([f"Question {i}:\n\"\"\"\n{qa['Q']}\n\"\"\"\n" \
                            f"Answer {i}:\n\"\"\"\n{qa['A']}\n\"\"\"" for i, qa in enumerate(qas)])
    text = f"Here are existing questions and their corresponding answers for the task:\n\n{qas_str}"
    prompt_his = get_user_prompt(text)
    text = "Please provide an comprehensive report to conclude the task."
    prompt_report = get_user_prompt(text)
    text, count = dsGPT([prompt_target, prompt_supp, prompt_issue, prompt_goal, 
                         prompt_his, prompt_report], OPENAI_MODEL)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_fname", "-t", type=str, required=True)
    parser.add_argument("--supp_fname", "-s", type=str, required=True)
    args = parser.parse_args()
    print(args)
    
    #issues = anomalyDetection(args.target_fname)    # TODO: Ask editGPT to extract issues
    issues = ["1. There is a sudden spike in the 'revenue' column on March 28th, 2023, with a value of 8.54 * 10^10, which exceeds the threshold limit of 3 times the standard deviation from the mean."]
    causeDiscovery(issues[0], args.target_fname, args.supp_fname, 3)
    
    print(f"\nCosting approx. NT. {TOKEN_COUNT / 4 / 1000 * 0.002 * 30} ({TOKEN_COUNT} characters) " 
           "as prompts to generate the summary.")
