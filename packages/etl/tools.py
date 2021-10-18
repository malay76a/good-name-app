import logging


# Insert df to table_name deleting outdated data by pk
def df_to_pg(df, table_name, schema, pk, engine, attempts=1):
    if pk is not None and pk.lower() == 'clear table':
        logging.info(f'All data will be deleted from {table_name}')
        delete_query = """DELETE FROM {table_name}""" \
            .format(table_name=schema + '.' + table_name)
    elif pk:
        logging.info(f'Data with be deleted from {table_name} by pk {pk}')
        delete_query = ("""
            DELETE FROM 
                {table_name} 
            WHERE 
                "{pk}" IN ('{pk_list}')
            """.format(
            pk=pk,
            pk_list="', '".join([str(i) for i in df[pk].unique()]),
            table_name=schema + '.' + table_name
        )
        )
    else:
        delete_query = None

    # INSERT
    for t in range(attempts):
        with engine.connect() as con:
            try:
                if delete_query:
                    logging.info(f'Deleting obsolete data from {schema}.{table_name}')
                    con.execute(delete_query)

                logging.info(f'Inserting to {schema}.{table_name}')
                df.to_sql(name=table_name, schema=schema, con=engine, if_exists='append', index=False, chunksize=5000)
                logging.info(f'Inserting to {schema}.{table_name} was successful!')
                break
            except Exception as e:
                logging.warning(f'Attempt {t + 1} failed. Reason: {e}')


def map_id(target_string: str, substring_to_id: dict):
    """
    Find in target string substring from substring_to_id keys and return its id
    :param target_string: any_string
    :param substring_to_id: {substing: unique_id}
    :return: id of matched substring
    """
    for key in substring_to_id.keys():
        if key[:-1] in target_string.lower().replace('.', ' '):  # [:-1] awesome crutch for extra whitespace in the end
            return substring_to_id[key]
