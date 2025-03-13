# Assuming it exists in graphrag.cli.query
from graphrag.cli.query import run_local_search
from pathlib import Path


# Prompt the user for input
user_input = input("Please enter your stream composition: ")

response, context_data = run_local_search(
    config_filepath=Path("./mw/settings.yaml"),
    data_dir=Path("./output"),
    root_dir=Path("./mw"),
    community_level=2,
    query=user_input,
    response_type="default",                      # If applicable
    streaming=False                                # If applicable
)




"""
query="Give a list of all the applications (APP), with the number of link(relationship or degree) per applications. From this list, enumerate the 10 applications with the most relationships in descending order of number of relationships starting with the one with the most relationships.",

query="Give a summary of the top 10 most popular applications (APP) in order of popularity starting with the most popular. Popularity is rated according to the number of different Quote IDs per application. For each application, give the number of degree (relationships) and report an exhaustive list of every single quotes. 'UV Detection Systems' and 'Community Analysis of Process Photometer ' are not applications.",
"""
