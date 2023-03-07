# Read data/housing-and-development-board-resale-price-index-1q2009-100-quarterly.csv
# which is in the format of:
# quarter,index
# 1990-Q1,24.3
# 1990-Q2,24.4
# 1990-Q3,25
# ...
# and replace each quarter with 3 months (e.g. 1990-Q1 becomes 1990-01, 1990-02, 1990-03, etc.)
# and write the result to data/housing-and-development-board-resale-price-index-1q2009-100-monthly.csv

import csv

with open('data/housing-and-development-board-resale-price-index-1q2009-100-quarterly.csv', 'r') as f:
    reader = csv.DictReader(f)
    with open('data/housing-and-development-board-resale-price-index-1q2009-100-monthly.csv', 'w') as f2:
        writer = csv.DictWriter(f2, fieldnames=['month', 'index'])
        writer.writeheader()
        for row in reader:
            year, quarter = row['quarter'].split('-Q')
            for month in range(0, 3):
                month = int(quarter) * 3 - 2 + month
                writer.writerow(
                    {'month': '{}-{:02d}'.format(year, month), 'index': row['index']})
