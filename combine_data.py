import pandas as pd
csv_paths = ["data/resale-flat-prices-based-on-approval-date-1990-1999.csv",
             "data/resale-flat-prices-based-on-approval-date-2000-feb-2012.csv",
             "data/resale-flat-prices-based-on-registration-date-from-mar-2012-to-dec-2014.csv",
             "data/resale-flat-prices-based-on-registration-date-from-jan-2015-to-dec-2016.csv",
             "data/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv",
             ]

# Combine CSV files into one while ignoring remaining_lease column
df = pd.concat((pd.read_csv(f, usecols=lambda x: x != 'remaining_lease')
               for f in csv_paths))

# Save combined CSV
df.to_csv("data/resale-flat-prices.csv", index=False)
