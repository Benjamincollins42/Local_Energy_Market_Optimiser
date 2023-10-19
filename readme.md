# Readme

## Introduction

In order to maximise the utility of a local energy market (LEM) a linear programming based approach has been developed, taking into account constraints on co2 production and selling and buying prices. This algorithm should find the solution which maximises the volume of power exchanged through the LEM. Additionally, the solution also contains suggested prices for these exchanges based on energy buying and selling price constraints.

Results of this algorithm on provided data have been then analysed along with the data. Metrics have been developed and graphs to assess performance.

An extension on this method is provided to allow for distance based restrictions on energy exchanges.

## Useage

After building the docker image with:

`docker-compose build`

you can run the matching algorithm on the provided data use:

`docker-compose up`

which will produce a .csv file for each time instance in the data, with the file going into `/transactions`.

## Transactions Output Format

Each valid transaction is output with the following details:
```
bid: id of the bid
to: id of the bid that power and money is being sent to
price_gbp: price to pay or be paid
volume_kwh: volume of power in kWh to be transfered
g_co2_produced: grams of co2 produced by the power generation
```
volume_kwh can be either positive or negative depending on if it is being brought or sold.

## Files

- #### `linear_prog_LEM_optimiser.py`
    - Set of functions used to comprise the developed matching algorithm. Provides functionality of use for bid data in the format of `bids.csv` with `-i` as input csv path and `-o` as output folder path.
- #### `matching_evaluation.py`
    - Set of functions used to visulise and assess the performance of the matching algorithm.
- #### `data_explore.ipynb`
    - Notebook investigating the provided data 
- #### `lat_lon_extension.ipynb`
    - Notebook expanding on key functions in `linear_prog_LEM_optimiser.py` to provide a solution for the use of latitude and longitude based distance restrictions on bids.
- #### `find_optimal_transactions.sh`
    - Bash script showing useage of the `linear_prog_LEM_optimiser.py` being run. Points to `/bids.csv` and `/transactions`.
- #### `requirements.txt`
    - Required python libaries.
- #### `Dockerfile`
    - Instructions to build the docker image.
- #### `docker-compose.yml`
    - Sets up the `docker-compose` functionality.
- #### `/data/assets.csv`
    - Provided details about the assets.
- #### `/data/bids.csv`
    - Provided set of bids for a local energy market.
- #### `/transactions/*`
    - Results of the matching algorithm for each time instance in `bids.csv`. detailing the transactions