import requests
import torch
import warnings
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from urllib.parse import urlparse
def fetch_github_repositories(url_entry):
    warnings.filterwarnings("ignore")
    global most_complex_repo, justification_text
     
    user_url = url_entry
    
    # Extracting the GitHub username from the user URL
    parsed_url = urlparse(user_url)
    username = parsed_url.path.strip("/")
    
    # Ensure the username is not empty
    if not username:
        print("Invalid GitHub user URL.")
        return
    
    # API endpoint to fetch user repositories
    api_url = f"https://api.github.com/users/{username}/repos"
    
    try:
        response = requests.get(api_url)

        # Raise exception if request was unsuccessful
        response.raise_for_status()
        repositories = response.json()

        # Extracting repository names
        repository_names = [repo['name'] for repo in repositories]
        
        # Initialize tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
  
        # Preprocess repository names
        def preprocess_repository_names(repository_names):
            for repo in repository_names:
                # Convert to lowercase and remove leading/trailing whitespaces
                repo = repo.strip().lower() 
                if repo: 
                    # Skip empty names
                    yield repo
                    
        # Process a repository
        def process_repository(repository_name):
            # Generate a prompt or template based on the repository name
            prompt = f"Evaluate the technical complexity of the repository: {repository_name}. Analyze the code and provide insights on its complexity."
            
            # Tokenize the prompt
            input_ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=100, return_tensors="pt")
            
            # Generate attention mask
            attention_mask = torch.ones_like(input_ids)

            # Generate output using the model
            with model.eval() and torch.no_grad():
                output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)

            # Decode the output
            processed_repo = tokenizer.decode(output[0], skip_special_tokens=True)

            return processed_repo
         
            
        preprocessed_names_generator = preprocess_repository_names(repository_names)
        
        
        # Process and score the preprocessed names
        repository_scores = {}
        for preprocessed_name in preprocessed_names_generator:
            processed_repo = process_repository(preprocessed_name)

            # The complexity score is based on the length of the processed repository which in turn is based on output of model.
            complexity_score = len(processed_repo)
            repository_scores[preprocessed_name] = complexity_score
        
        # Identify the repository with the highest complexity score
        most_complex_repo = max(repository_scores, key=repository_scores.get)
        
        # Justify the selection using GPT
        justification_prompt = f"Justification for selecting the most technically complex repository: {most_complex_repo}."
        justification_input_ids = tokenizer.encode(justification_prompt, add_special_tokens=False, truncation=True, max_length=100, return_tensors="pt")
        
        attention_mask = torch.ones_like(justification_input_ids)
        
        with torch.no_grad():
            justification_output = model.generate(input_ids=justification_input_ids, attention_mask=attention_mask, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
        
        justification_text = tokenizer.decode(justification_output[0], skip_special_tokens=True)

        return most_complex_repo,justification_text
        
        
    except requests.exceptions.HTTPError as err:
        print(f"An HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}")        
        
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods =["GET","POST"])
def gfg():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       url_entry = request.form.get("first_name")
       fetch_github_repositories(url_entry)
       return render_template("ans.html", most_complex_repo=most_complex_repo,justification_text=justification_text, repo_url=url_entry+'/'+most_complex_repo)
    return render_template("index.html")

if __name__ == '__main__':
    app.run()
