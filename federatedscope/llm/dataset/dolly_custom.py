import os
import json
from federatedscope.llm.dataloader.dataloader import load_jsonl


def main():
    
    file_path_original = "data/databricks-dolly-15k_original.jsonl"
    file_path_train = "data/databricks-dolly-15k.jsonl"
    file_path_eval = "data/databricks-dolly-15k_eval.jsonl"

    if not os.path.exists(file_path_original):
        print("Error, no local dolly dataset.")

    else:
        # Load original data once
        original_data_dict = load_jsonl(file_path_original,
                                instruction='instruction',
                                input='context',
                                output='response',
                                category='category')

        # Get train file, with every task other that summarization
        if not os.path.exists(file_path_train):
            train_data_dict = [
                x for x in original_data_dict if x["category"] != "summarization"
            ]    

            with open(file_path_train, 'w') as f:
                for item in train_data_dict:
                    json_line = json.dumps(item)
                    f.write(json_line + '\n')
                print("Saved training dolly dataset.")
        
        # Get evaluation file, with summarization task
        if not os.path.exists(file_path_eval):
            eval_data_dict = [
                x for x in original_data_dict if x["category"] == "summarization"
            ]    

            with open(file_path_eval, 'w') as f:
                for item in eval_data_dict:
                    json_line = json.dumps(item)
                    f.write(json_line + '\n')
                print("Saved evaluation dolly dataset.")

if __name__ == '__main__':
    main()