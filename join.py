from pathlib import Path

import pandas as pd



ACCOUNTS_FILE =r"C:\Users\Samarth\Downloads\HI-Small_accounts.csv"
TRANSACTIONS_FILE = r"C:\Users\Samarth\Downloads\HI-Small_Trans.csv"


def resolve_transaction_account_column(frame: pd.DataFrame) -> str:
    for column_name in ("Account", "Account.1"):
        if column_name in frame.columns:
            return column_name
    raise KeyError("Could not find an account column in the transactions file.")


accounts = pd.read_csv(ACCOUNTS_FILE)
transactions = pd.read_csv(TRANSACTIONS_FILE)

transaction_account_column = resolve_transaction_account_column(transactions)

df_join = transactions.merge(
    accounts,
    left_on=transaction_account_column,
    right_on="Account Number",
    how="inner",
)

print(df_join.shape)
print(df_join.head())

df_join.to_csv( r"C:\Users\Samarth\Downloads\joined_data.csv", index=False)