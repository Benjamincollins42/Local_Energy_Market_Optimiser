"""Module providing metrics and visulisation for LEM optimising performance."""

import matplotlib.pyplot as plt
import pandas as pd


def evaluate_solution_power(transactions_df: pd.DataFrame, bid_df: pd.DataFrame) -> float:
    """Calulates the remaining power of a solution compared to the constraintless optimum

    Parameters
    ----------
    transactions_df : pandas.DataFrame
        Dataframe containing all the transactions required to optimise the utility of a time period
    bid_df : pandas.DataFrame
        Dataframe containing all bids for a time period

    Returns
    -------
    float
        Remaining power of a solution compared to the constraintless optimum
    """
    power_throughfair = sum(abs(transactions_df.volume_kwh))

    constraintless_throughfair_max = sum(abs(bid_df.volume_kwh))- abs(bid_df.volume_kwh.sum())

    power_loss = constraintless_throughfair_max - power_throughfair

    return power_loss


def evaluate_solution_power_relative(transactions_df: pd.DataFrame, bid_df: pd.DataFrame) -> float:
    """Calulates the normalised remaining power of a solution compared to the constraintless optimum

    Parameters
    ----------
    transactions_df : pandas.DataFrame
        Dataframe containing all the transactions required to optimise the utility of a time period
    bid_df : pandas.DataFrame
        Dataframe containing all bids for a time period


    Returns
    -------
    float
        Normalised remaining power of a solution compared to the constraintless optimum
    """
    power_throughfair = sum(abs(transactions_df.volume_kwh))

    constraintless_throughfair_max = sum(abs(bid_df.volume_kwh))- abs(bid_df.volume_kwh.sum())

    power_loss = constraintless_throughfair_max - power_throughfair

    return power_loss / constraintless_throughfair_max


def evaluate_solution_cashflow(transactions_df: pd.DataFrame) -> float:
    """Calculates the amount of cashflow for each transaction solution

    Parameters
    ----------
    transactions_df : pandas.DataFrame
        Dataframe containing all the transactions required to optimise the utility of a time period

    Returns
    -------
    float
        Amount of cashflow for transaction solution
    """
    return round(transactions_df.price_gbp.sum()/2,2)


def evaluate_solution_bidder_penetration(transactions_df: pd.DataFrame, bid_df: pd.DataFrame) -> float:
    """Calculates the % of bidders with a valid transaction

    Parameters
    ----------
    transactions_df : pandas.DataFrame
        Dataframe containing all the transactions required to optimise the utility of a time period
    bid_df : pandas.DataFrame
        Dataframe containing all bids for a time period

    Returns
    -------
    float
        Percentage of bidders with a valid transaction
    """
    return len(transactions_df.bid.unique())/len(bid_df.bid.unique())


def plot_power_evaluation(power_evaluation: list[float]) -> None:
    """Generates two pyplot graphs showing the power evaluation

    Parameters
    ----------
    power_evaluation : list(float)
        Remaining power of a solution compared to the constraintless optimum for each time period
    """
    plt.figure(figsize=(8,6),dpi=100)
    plt.plot(range(len(power_evaluation)), power_evaluation)
    plt.xlabel('Time Slot (Hour)')
    plt.ylabel('Loss Potential Constraintless Power (kWh)')
    plt.show()

    plt.figure(figsize=(8,6),dpi=100)
    plt.hist(power_evaluation,bins = 20)
    plt.xlabel('Loss Potential Constraintless Power kWh')
    plt.ylabel('Time Instance Count')
    plt.show()


def plot_cash_evaluation(cash_evaluation: list[float]) -> None:
    """Generates two pyplot graphs showing the cash flow evaluation

    Parameters
    ----------
    cash_evaluation : list(float)
        Amount of cashflow for each transaction solution
    """
    plt.figure(figsize=(8,6),dpi=100)
    plt.plot(range(len(cash_evaluation)), cash_evaluation)
    plt.xlabel('Time Slot (Hour)')
    plt.ylabel('Cash Flow (£)')
    plt.show()


def plot_user_evaluation(user_evaluation: list[float]) -> None:
    """Generates two pyplot graphs showing the user evaluation

    Parameters
    ----------
    user_evaluation : list(float)
        Percentage of bidders with a valid transaction for each transaction solution
    """
    plt.figure(figsize=(8,6),dpi=100)
    plt.plot(range(len(user_evaluation)), user_evaluation)
    plt.xlabel('Time Slot (Hour)')
    plt.ylabel('Customer Penetration (%)')
    plt.show()
