"""
### The sentiment of Manatee Jokes
"""

from airflow.decorators import dag
from astro import sql as aql
from astro.sql.table import Table
from airflow.operators.empty import EmptyOperator

import pandas as pd
from pendulum import datetime
import logging
import requests
import os

task_logger = logging.getLogger("airflow.task")

DB_CONN_ID = "snowflake_default"


@aql.dataframe
def get_sentences_from_api():
    r = requests.get("https://manateejokesapi.herokuapp.com/manatees/random")
    df = pd.json_normalize(r.json())
    df.columns = [col_name.upper() for col_name in df.columns]
    df = df.rename(columns={"ID": "JOKE_ID"})
    print(df)
    return df


@aql.transform
def transform(in_table):
    return """
            SELECT SETUP, PUNCHLINE
            FROM {{ in_table }};
            """


@aql.dataframe
def sentiment_analysis(df: pd.DataFrame):
    print(df)
    api_token = os.environ["API_TOKEN"]
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    headers = {"Authorization": f"Bearer {api_token}"}

    query = list(df["setup"].values) + list(df["punchline"].values)
    print(query)

    api_url = f"https://api-inference.huggingface.co/models/{model_name}"

    response = requests.post(api_url, headers=headers, json={"inputs": query})

    if response.status_code == 200:
        response_text = response.json()
        task_logger.info(response_text)
    else:
        task_logger.info(
            f"Request failed with status code {response.status_code}: {response.text}"
        )


@dag(
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
)
def live_pipeline():
    start = EmptyOperator(task_id="start")

    in_data = get_sentences_from_api(
        output_table=Table(conn_id=DB_CONN_ID, name="in_joke")
    )

    transformed_data = transform(
        in_table=in_data, output_table=Table(conn_id=DB_CONN_ID, name="joke_table")
    )

    validate_table = aql.check_table(
        dataset=transformed_data,
        checks={
            "row_count": {"check_statement": "Count(*) >= 1"},
        },
    )

    validate_columns = aql.check_column(
        dataset=Table(conn_id=DB_CONN_ID, name="joke_table"),
        column_mapping={
            "setup": {"null_check": {"equal_to": 0}},
            "punchline": {"null_check": {"equal_to": 0}},
        },
    )

    run_model = sentiment_analysis(transformed_data)

    transformed_data >> [validate_table, validate_columns] >> run_model

    aql.cleanup()


live_pipeline()
