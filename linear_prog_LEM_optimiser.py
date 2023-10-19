"""Module containing a linear programming based LEM optimiser."""

import pandas as pd
import pulp as pl
import numpy as np


def prepare_data(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """Reformats initial Dataframe into buyers and sellers 

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe for a period of time of buyer and seller bids

    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
        Dataframe containing all the sell bids, Dataframe containing all the buy bids
    """
    df = df.copy()
    # pulling out the price data into a more usable format
    df['price_gbp_kwh_sign'] = df['price_gbp_kwh'].str[0]
    df['price_gbp_kwh_value'] = df['price_gbp_kwh'].str[1:].astype(float)

    seller_df = df[df.volume_kwh < 0].reset_index()
    buyer_df = df[df.volume_kwh > 0].reset_index()

    # pulling out the co2 data into a more usable format
    buyer_df['carbon_gco2_kwh_sign'] = buyer_df['carbon_gco2_kwh'].str[0].fillna('>')
    buyer_df['carbon_gco2_kwh_value'] = buyer_df['carbon_gco2_kwh'].str[1:].astype(float).fillna(0)

    return seller_df, buyer_df


def get_valid_purchases(seller_df: pd.DataFrame, buyer_df: pd.DataFrame) -> np.array(int):
    """Determine which purchases are valid between seller and buyer pairs

    Parameters
    ----------
    seller_df : pandas.DataFrame
        Dataframe containing all the sell bids
    buyer_df : pandas.DataFrame
        Dataframe containing all the buy bids

    Returns
    -------
    numpy.array(int)
        2D array encoding valid purchases with a 1 indexed [buyer, seller]
    """
    valid_purchase = np.zeros((len(buyer_df), len(seller_df)), dtype='uint8')
    valid_sell = np.zeros((len(buyer_df), len(seller_df)), dtype='uint8')


    for i, sell_bid in enumerate(seller_df.itertuples(index=False)):
        # finding all valid purchases for each sell bid
        valid_buys_index = buyer_df.query(f'`price_gbp_kwh_value`{sell_bid.price_gbp_kwh}').index
        valid_purchase[valid_buys_index, i] = 1

    for i, buy_bid in enumerate(buyer_df.itertuples(index=False)):
        # finding all valid purchases for each buy bid
        valid_sells_index = seller_df.query(f'`price_gbp_kwh_value`{buy_bid.price_gbp_kwh}').index
        valid_sell[i, valid_sells_index] = 1

    return valid_purchase * valid_sell



def generate_variables(seller_df: pd.DataFrame, buyer_df: pd.DataFrame, valid_purchase: np.array(int)) -> np.array(pl.LpVariable):
    """Generates the required purchase variables for the linear programming problem

    Parameters
    ----------
    seller_df : pandas.DataFrame
        Dataframe containing all the sell bids
    buyer_df : pandas.DataFrame
        Dataframe containing all the buy bids
    valid_purchase : numpy.array
        2D array encoding valid purchases with a 1 indexed [buyer, seller]

    Returns
    -------
    np.array(pulp.LpVariable)
        2D array of Pulp LpVariables which encode purchase amount to be optimised for each valid sellery buyer pair
    """
    valid_buys, valid_sells = np.where(valid_purchase == 1)

    # each purchase variable is amount of sell bid comsumed
    purchase_variables = np.zeros(shape=(len(seller_df),len(buyer_df)), dtype=pl.LpVariable)

    for sell_index, buy_index in zip(valid_sells,valid_buys):
        purchase_variables[sell_index, buy_index] = pl.LpVariable(f"x_{sell_index}_{buy_index}", lowBound=0, upBound=1)
    
    return purchase_variables


def set_selling_constraints(lp_problem: pl.LpProblem, purchase_variables: np.array(pl.LpVariable)) -> pl.LpProblem:
    """Generates the required constraints for the linear programming problem for the selling bids

    Parameters
    ----------
    lp_problem : pulp.LpProblem
        The linear programming problem
    purchase_variables : numpy.array(pulp.LpVariable)
        2D array of Pulp LpVariables which encode purchase amount to be optimised for each valid sellery buyer pair

    Returns
    -------
    lp_problem : pulp.LpProblem
        The updated linear programming problem
    """
    # Each sale can only use its total volume
    for sell_index in purchase_variables:
        if sum(sell_index) != 0:
            lp_problem += pl.lpSum(sell_index) <= 1

    return lp_problem


def set_buying_constraints(lp_problem: pl.LpProblem, purchase_variables: np.array(pl.LpAffineExpression),
                           b_volume: np.array(float), s_volume: np.array(float), gco2_rate: np.array(float),
                           buyer_df: pd.DataFrame) -> (pl.LpProblem, list[pl.LpAffineExpression]):
    """Generates the required constraints for the linear programming problem for the buying bids

    Parameters
    ----------
    lp_problem : pulp.LpProblem
        The linear programming problem
    purchase_variables : numpy.array(pulp.LpVariable)
        2D array of Pulp LpVariables which encode purchase amount to be optimised for each valid sellery buyer pair
    b_volume : numpy.array(float)
        List of all the buyer power volume quanities
    s_volume : numpy.array(float)
        List of all the seller power volume quanities
    gco2_rate : numpy.array(float)
        List of all the seller grams of co2 per kWh quanities
    buyer_df : pandas.DataFrame
        Dataframe containing all the buy bids

    Returns
    -------
    pulp.LpProblem, list(pulp.LpAffineExpression)
        The updated linear programming problem, list of linear programming buying conditions
    """
    buying_conditions = []

    for buy_bid_index in range(len(b_volume)):

        # constraints such that you can only supply what is wanted
        constraint_expression = pl.lpSum(
            purchase_variables[sell_bid_index, buy_bid_index] * s_volume[sell_bid_index]
            for sell_bid_index in range(len(s_volume))
            ) + b_volume[buy_bid_index]
        buying_conditions.append(constraint_expression)
        lp_problem += constraint_expression >= 0
        
        # co2_condtions for each buy bid
        lp_problem = set_co2_contraints(lp_problem, buyer_df, buy_bid_index, purchase_variables, s_volume, gco2_rate)

    return lp_problem, buying_conditions


def set_co2_contraints(lp_problem: pl.LpProblem, buyer_df: pd.DataFrame, buy_bid_index: int, purchase_variables: np.array(pl.LpVariable),
                       s_volume: np.array(float), gco2_rate: np.array(float)) -> pl.LpProblem:
    """Generates the required constraints for the linear programming problem for the selling bids in terms of co2

    Parameters
    ----------
    lp_problem : pulp.LpProblem
        The linear programming problem
    buyer_df : pandas.DataFrame
        Dataframe containing all the buy bids
    buy_bid_index : int
        Index of the buy bid constraints are being generated for
    purchase_variables : numpy.array(pulp.LpVariable)
        A 2D array of Pulp LpVariables which encode purchase amount to be optimised for each valid sellery buyer pair
    s_volume : numpy.array(float)
        List of all the seller power volume quanities
    gco2_rate : numpy.array(float)
        List of all the seller grams of co2 per kWh quanities

    Returns
    -------
    pulp.LpProblem, list(pulp.LpAffineExpression)
        The updated linear programming problem, list of linear programming buying conditions
    """
    if not pd.isna(buyer_df.iloc[buy_bid_index].carbon_gco2_kwh):

        buyer_carbon_value = buyer_df.iloc[buy_bid_index].carbon_gco2_kwh_value
        buyer_carbon_sign = buyer_df.iloc[buy_bid_index].carbon_gco2_kwh_sign
        
        # total volume of co2 generated compared the the value of the co2 generated constraint
        constraint_expression = pl.lpSum(
            purchase_variables[sell_bid_index, buy_bid_index] * -s_volume[sell_bid_index] *
            (gco2_rate[sell_bid_index] - buyer_carbon_value) for sell_bid_index in range(len(s_volume))
        )
        
        if buyer_carbon_sign == '<':
            lp_problem += constraint_expression <= 0
        elif buyer_carbon_sign == '>':
            lp_problem += constraint_expression >= 0
        elif buyer_carbon_sign == '=':
            lp_problem += constraint_expression == 0

    return lp_problem


def calculate_sale_price(buy_bid: pd.Series, sell_bid: pd.Series) -> float:
    """Calculates the sale price between a buy_bid and sell_bid

    Parameters
    ----------
    buy_bid : pandas.Series
        Data about a specific buying bid
    sell_bid : pandas.Series
        Data about a specific selling bid

    Returns
    -------
    float
        Sale price in gbp

    Raises
    ------
    Exception
        Unaccounted for sell buy case
    """
    # for readability
    buy_value, buy_sign = buy_bid['price_gbp_kwh_value'], buy_bid['price_gbp_kwh_sign']
    sell_value, sell_sign = sell_bid['price_gbp_kwh_value'], sell_bid['price_gbp_kwh_sign']

    # sale pricing can be modified

    # same sign conditions
    if buy_sign == sell_sign:
        if buy_sign == '>':
            sale_price = max(buy_value, sell_value)
        elif buy_sign == '<':
            sale_price = min(buy_value, sell_value)
        else:
            sale_price = buy_value if buy_value == sell_value else None
    # different sign conditions
    else:
        if (buy_sign, sell_sign) in [('>', '<'), ('<', '>')]:
            sale_price = (buy_value + sell_value) / 2
        elif buy_sign == '=':
            sale_price = buy_value
        elif sell_sign == '=':
            sale_price = sell_value

    if sale_price is None:
        raise Exception(f'Case not accounted for {buy_bid.bid}, {sell_bid.bid}')

    return round(sale_price, 2)


def process_linear_programming_solution(lp_problem: pl.LpProblem, buyer_df: pd.DataFrame, seller_df: pd.DataFrame,
                                        s_volume: np.array(float), gco2_rate: np.array(float)) -> pd.DataFrame:
    """Processes the linear programming solution to extract optimal transactions

    Parameters
    ----------
    lp_problem : pulp.LpProblem
        Solved linear programming problem
    seller_df : pandas.DataFrame
        Dataframe containing all the sell bids
    buyer_df : pandas.DataFrame
        Dataframe containing all the buy bids
    s_volume : numpy.array(float)
        List of all the seller power volume quanities
    gco2_rate : numpy.array(float)
        List of all the seller grams of co2 per kWh quanities

    Returns
    -------
    pandas.DataFrame
        Dataframe containing all the transactions required to optimise the utility of a time period

    Raises
    ------
    Exception
        No valid solution found
    """
    transactions = []
    if pl.LpStatus[lp_problem.status] == "Optimal":
        
        for var in lp_problem.variablesDict().values():
            if var.varValue != 0:
                buy_bid_index = int(var.name.split('_')[2])
                sell_bid_index = int(var.name.split('_')[1])
                sale_price = calculate_sale_price(buyer_df.iloc[buy_bid_index], seller_df.iloc[sell_bid_index])
                co2_produced = round(var.varValue*gco2_rate[sell_bid_index]*-s_volume[sell_bid_index], 3)
                # transaction to buy
                transactions.append(
                    {'bid': buyer_df.iloc[buy_bid_index].bid,
                     'to': seller_df.iloc[sell_bid_index].bid,
                     'price_gbp': sale_price,
                     'volume_kwh': round(-var.varValue*s_volume[sell_bid_index], 3),
                     'g_co2_produced': co2_produced})
                # transaction to sell
                transactions.append(
                    {'bid': seller_df.iloc[sell_bid_index].bid,
                     'to': buyer_df.iloc[buy_bid_index].bid,
                     'price_gbp': sale_price,
                     'volume_kwh': round(var.varValue*s_volume[sell_bid_index], 3),
                     'g_co2_produced': co2_produced})
    else:
        raise Exception("No optimal solution found")
    
    return pd.DataFrame.from_dict(transactions)


def linear_programming_matcher(df: pd.DataFrame) -> pd.DataFrame:
    """Algorithm to match buyer and seller bids of electricty to optimise the utility of a time period.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe for a period of time of buyer and seller bids

    Returns
    -------
    pandas.DataFrame
        Dataframe containing all the transactions required to optimise the utility of a time period
    """
    # preparing data for linear programming
    seller_df, buyer_df = prepare_data(df)
    valid_purchase = get_valid_purchases(seller_df, buyer_df)
    purchase_variables = generate_variables(seller_df, buyer_df, valid_purchase)

    # minimisation problem
    lp_problem = pl.LpProblem(name='linear_problem', sense=1)

    # Define the coefficients lists
    b_volume = np.array(buyer_df.volume_kwh.values)
    s_volume = np.array(seller_df.volume_kwh.values)
    gco2_rate = np.array(seller_df.carbon_gco2_kwh.replace(np.nan,0).values.astype(float))

    # setting constraints
    lp_problem = set_selling_constraints(lp_problem, purchase_variables)
    lp_problem, buying_conditions = set_buying_constraints(lp_problem, purchase_variables, b_volume, s_volume, gco2_rate, buyer_df)

    # setting the objective function
    lp_problem += pl.lpSum(buying_conditions)

    lp_problem.solve()

    # processing the linear problem
    transactions = process_linear_programming_solution(lp_problem, buyer_df, seller_df, s_volume, gco2_rate)
    
    return transactions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '-input_bid_csv_path', help="Path to .csv file containing bids", required=True)
    parser.add_argument('-o', '-output_path', help="Output folder path", required=True)
    args = parser.parse_args()
    config = vars(args)

    bids_df = pd.read_csv(f'{config["i"]}', index_col=0)

    # run the matching algorithm on each unique time instance
    for time in bids_df.time.unique():
        working_df = bids_df.query(f'`time` == {time}')
        working_transactions = linear_programming_matcher(working_df)
        working_transactions.to_csv(f'{config["o"]}/transactions_at_{time}.csv')
