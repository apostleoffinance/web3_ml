# python-template
 A conda based template for easy setup of reproducible repos integrating Flipside data & Python for data science.

 We recommend the lightweight miniconda: https://docs.conda.io/projects/miniconda/en/latest/ alongside VSCode with Jupyter & Python plugins.

# Use as a Template for a new Repo

~10 min video review of this process available here: https://www.loom.com/share/Using-the-Python-Template-for-Reproducible-Data-Science-2c524ce317394dc7884cc8927b524910
1. Clone this repo.
2. Change the folder name to your desired repo name.
3. Inside the  `environment.yml` there is a name parameter, rename it to your desired repo name.
4. delete the (possibly hidden) .git folder to remove the commit history.
5. Create a git repo for this folder (e.g., using GitHub Desktop). You can keep the same gitignore, license, README as the template or replace them. But if you replace, ensure you protect your API keys.
6. Open miniconda and navigate to your folder `cd ...Documents/yourNewFolderName`
7. Restore the conda environment `conda env create -f environment.yml` this will use the new name you placed in the environment.yml file.
8. Activate the environment `conda activate yourEnvName`
9. Open the folder in VS Code.
10. Create & place an api_key.txt file with your key in the folder (note: ensure it is gitignored).
11. You can now run hello-flipside.py interactively (ensure you switch to your Conda yourEnvName Kernel to access the packages).
12. Use  `conda env export > environment.yml` to add new packages as you work from the template & your new repo.

# Reproduce 

1. Clone this repo. I use GitHub Desktop, you can grab the URL to clone in that app, or use `git clone https://github.com/fsc-data-science/python-template.git`
2. Create a txt file called `api_key.txt` and place your API key (you can create one via the API page when logged into Flipside: https://flipsidecrypto.xyz/account/api-keys )
3. Open miniconda3  and move to your cloned directory `cd YourLocationHere/python-template`
4. In Conda create a new environment using the provided environment.yml file: `conda env create -f environment.yml` and optionally rename with `--name new_conda_name`

The environment yml includes installs for flipside, numpy, pandas, and plotly, alongside this python-template environment name and python 3.10.9

5. Activate the new environment: `conda activate python-template` 
6. Run the analysis script `hello-flipside.py` using your `python-template` Kernel. If you create an interactive HTML file (`candlestick_chart.html`) it worked! (Open in browser to see the eth volume weighted average price candle chart over the last 30 days).
7. You can restart your git history, rename the directory, and/or clone & replace the conda with `conda create --name new_environment_name --clone python-template` 

8. As you work with your own repos and environments, use `conda env export > environment.yml` to manage new package installations and ensure reproducibility for those who clone your repos!

# Flipside API 

You can generate API keys to bring SQL queries from the data studio into Python 
via the flipside package. This repo includes it in the environment for you. Note: API Keys are capped at a free 5,000 query seconds/month.
 So it is recommended you write your queries in the (free) Studio first, and then bring the polished query into Python once it runs as expected.

Relatedly, keep our docs handy as you work with data from Flipside. 

https://docs.flipsidecrypto.com/flipside-api/get-started/run-your-first-query

For simplicity you may want to keep this auto-paginate function handy until it is formally added to the package. You can convert the result to a pandas data frame using `pd.DataFrame()`

```
"""This function will be added to Flipside package after testing, just copy/paste as needed for now"""
def auto_paginate_result(query_result_set, page_size=10000):
    """
    This function auto-paginates a query result to get all the data. It assumes 10,000 rows per page.
    In case of an error, reduce the page size. Uses numpy.
    """
    num_rows = query_result_set.page.totalRows
    page_count = np.ceil(num_rows / page_size).astype(int)
    all_rows = []
    current_page = 1
    while current_page <= page_count:
        results = flipside.get_query_results(
            query_result_set.query_id,
            page_number=current_page,
            page_size=page_size
        )

        if results.records:
            all_rows.extend(results.records)  # Use extend() to add list elements

        current_page += 1  # Increment the current page number

    return all_rows  # Return all_rows in JSON format
```

