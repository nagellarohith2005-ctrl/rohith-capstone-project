# Data Pipeline

## Overview

This module implements a complete data pipeline for collecting, cleaning, transforming, and storing book catalog data.

The pipeline collects book information from a public scraping-practice website, cleans the collected data, converts prices from GBP to INR using a fixed exchange rate, and stores the processed data in a normalized SQLite database.

## Data Source

The data is collected from:

Books to Scrape

The pipeline collects book-level information such as:

- Book title
- Price in GBP
- Star rating
- Availability
- Category

## Data Processing

The scraped data is cleaned and transformed by:

- Removing unnecessary characters from prices
- Converting prices to numeric values
- Converting star ratings into integers
- Converting availability into a Boolean value
- Handling invalid or missing values
- Converting GBP prices into INR

The fixed exchange rate used is:

1 GBP = 105.50 INR

## Database

The cleaned data is stored in a SQLite database:

`books.db`

The database is organized using relational tables for books and categories.

## SQL Analysis

SQL queries are used to analyze the stored data, including:

- Book counts
- Category-wise information
- Price analysis
- Rating analysis
- Availability analysis

## Technologies Used

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite
- SQL
- Jupyter Notebook / Google Colab

## Files

| File | Description |
|---|---|
| `data_pipeline.ipynb` | Complete data scraping, cleaning, transformation, and database pipeline |
| `books.db` | SQLite database containing processed book data |
| `README.md` | Documentation for Module 1 |

## How to Run

1. Open `data_pipeline.ipynb`.
2. Install the required Python libraries if necessary.
3. Run the notebook cells in order.
4. The pipeline scrapes the data.
5. The data is cleaned and transformed.
6. Prices are converted from GBP to INR.
7. The processed data is stored in the SQLite database.
8. SQL queries are executed for analysis.

## Module Outcome

The completed pipeline demonstrates the process of transforming raw web data into clean, structured, queryable data that can be used for further analysis and reporting.
