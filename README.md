# V2G Economic Feasibility under Dynamic Pricing and Demand Response Programs

[insert DOI]


## Abstract
 
Vehicle-to-grid (V2G) charging is a promising grid asset that is currently hampered by concerns of high capital costs and battery degradation. V2G simulations have demonstrated profitability under static electricity prices and ancillary market participation. However, it is unknown whether passenger V2G is economically viable with participation in nascent dynamic pricing and demand response programs available for consumers in 2025. Furthermore, it has not been determined which types of drivers stand to financially benefit the most from these programs. In a case study of California's PG&E utility territory, we simulate annual and net present costs over a vehicle's lifetime with V2G participation in dynamic pricing and demand response programs, testing sensitivity to charger access, installation costs, and battery degradation. 
With high charger access, favorable dynamic prices, and low installation costs, V2G can yield a profit of up to $1,181 annually per vehicle with a 4.5 year break-even time for the charger and installation costs. However, if these criteria are not met, the system may take between 5-15 years to pay itself off.
These findings suggest that V2G adoption should be limited to only specific groups of drivers with certain dynamic price structures, unless dynamic prices change or installation costs decrease in the future grid.

## About

Please contact Sonia Martin (soniamartin@stanford.edu) with questions about this repository. 

Research team: Sonia Martin (Stanford), William A. Paxton (Volkswagen), and Ram Rajagopal (Stanford) 

This research project was funded by Volkswagen Group of America.

This code accompanies a paper submitted to Joule entitled "Residential Vehicle-to-Grid Economics under Dynamic Pricing: The Role of Price Variation and Real-World Charging Behavior".

## License 

This code is licensed under the CC BY-NC-SA 4.0 license. The legal code can be found here: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en

## Code

### Initializing Python Environment

Windows PC instructions:
Instructions are provided to create a virtual environment created with .venv in Visual Studio Code on a Windows PC.

1) Download Python 3.10 or 3.11 at https://www.python.org/downloads/
2) Download VS Code with build tools at https://code.visualstudio.com/download (works for Windows or Mac)
3) Install the Python and Jupyter extensions on VS Code.
4) To create a new virtual environment on VS Code, press View -> Command Palette and search for Python:Create New Environment. Click on Venv for the .venv virtual environment. 
5) Activate the virtual environment by running this command in Windows cmd: 'your_path\your_venv_name\Scripts\activate.bat'
5) Download the correct package versions by running the command below in the terminal:

pip install -r requirements.txt


### Optimization Solver

Optimization with CVXPY is run with the MOSEK solver. The license is available for free for academic users and offers a 30 day free trial for private users. Please see https://www.mosek.com/resources/getting-started/ to download a license. The .lic file from MOSEK must be stored in the correct folder for CVXPY to correctly run. There is a code block that will print an error message if the license is not present. 

### Structure

This repository contains four folders: 
1. Hourly_Prices
2. Parameters 
3. Vehicle Data

PG&E hourly dynamic prices for each of the four circuits tested are located in the Hourly_Prices Folder

The Parameters folder contains the configuration file for each circuit with and without a battery aging cost.
 
Vehicle Data contains the raw sample vehicle data. 

The main repository contains have the three main model classes: 
1. charging.py
2. clustering.py
3. optimization.py

The run_simulation.ipynb file includes code cells to to create the baseline data, create the feature array, run the clustering, and run the optimization.

### Data

In this repository, a sample dataset is provided to run the code; the values are generated and not based on real EV traces. The full raw EV dataset used in the paper submission is located at https://data.mendeley.com/datasets/dszk8rzfjx/1. To run the code with the full dataset, change the filename of the input data file in the first code block in run_simulation.ipynb. Also, update "min_date_data" and "max_date_data" in the params JSON files to align with the dates present in the input data (this can be multiple months if the input data files are concatenated).

### Running Instructions

Run all code cells in run_simulation.ipynb. Creating the baseline data and clustering only need to be run once.

Note that the Mosek check is important to ensure correct results. (The optimization code will not throw an error if the Mosek file is missing, so you should confirm with the code block check instead.)

To run the figure notebooks, which are labeled as they appear in the paper, ensure the results are stored in the Results folder, which is created upon running the optimization simulation.

