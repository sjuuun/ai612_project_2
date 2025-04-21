# AI612 Project 2: Conversational Text-to-SQL Agent

The goal of this project is to implement a conversational text-to-SQL agent (DB Agent) for EHRs that can interact with both a user and a database. To simulate user interactions, you will be provided with a User LLM mimicking a human user that asks questions about the data stored in the EHR database (MIMIC-IV Demo). When interacting with the user, it should interact in natural language, and when interacting with the database, it should interact by using appropriate tools tailored to the database. After gathering all necessary information from the user and the database, the DB agent must generate a valid SQL to retrieve the final answer.

Furthermore, it is guaranteed that all questions asked by the User LLM are answerable. As a result, for each conversation initiated by the User LLM, your DB Agent should generate a correct SQL query. Additionally, note that values mentioned in natural language questions may use different phrasing compared to their representation in the database (e.g., "Hb" vs. "hemoglobin"). Addressing this requires the DB Agent to leverage appropriate tools to explore the database and identify the correct database entities or values corresponding to the user's input.


Check [Project 2 Specs](https://docs.google.com/document/d/18SVb7a7R0UedJabTadoqJrc1O_-29a_zEEX1AAZ0R2s/edit?usp=sharing) for more details.

## Database
We use the [MIMIC-IV database demo](https://physionet.org/content/mimic-iv-demo/2.2/), which anyone can access the files as long as they conform to the terms of the [Open Data Commons Open Database License v1.0](https://physionet.org/content/mimic-iv-demo/view-license/2.2/).

## Tutorials
Check out the jupyter notebook file [`tutorial.ipynb`](tutorial.ipynb). It includes an implementation of a baseline tool-calling DB Agent and how to run inference for submission. The score for the baseline agent on the provided validation set is 40 and costs about $0.20.


## Dependencies
The codes were tested in Colab and in a local environment (Python 3.11) with the dependecies listed in `requirements.txt`.

### Local
```bash
conda create -n ai612-project2 python=3.11 -y
conda activate ai612-project2
pip install -r requirements.txt
```

### How to run
```bash
sh run_mimic_iv.sh # running inference
streamlit run visualizer.py --server.port 8505 # visualizing conversations
```

### Tau-bench
This code repository is based on the [tau-bench github repository](https://github.com/sierra-research/tau-bench.git).