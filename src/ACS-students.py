# 2024 ACS 5-year
# find county level counts of current undergrads and grad students

import requests
import pandas as pd
import numpy as np


# insert Census API key here
API_KEY = "INSERT API KEY"

# config
YEAR = "2024"
STATE_FIPS = "51"  # Virginia
OUTPUT_CSV = "outputs/acs_students.csv"

# ACS variables to call
VARIABLES = [
    "B11001_001E",  # Total households (estimate)
    "B11001_001M",   # Total households (MOE)

    "B01003_001E",  # Total population (estimate)
    "B01003_001M",  # Total population (MOE)

    "B14001_008E",  # Undergrad enrolled in college (estimate)
    "B14001_008M",  # Undergrad enrolled in college (MOE
    "B14001_009E",  # Grad students enrolled in college (estimate)
    "B14001_009M",  # Grad students enrolled in college (MOE


]

# Census API endpoint
url = f"https://api.census.gov/data/2024/acs/acs5"


params = {
    "get": ",".join(VARIABLES),
    "for": "county:*",
    "in": f"state:{STATE_FIPS}",
    "key": API_KEY
}


# data request
response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

# create dataframe
df = pd.DataFrame(data[1:], columns=data[0])

# create tract GEOID
df["TRACT_GEOID"] = df["state"] + df["county"]

# clean
df["HH"] = pd.to_numeric(df["B11001_001E"], errors="coerce")
df["HH_MOE"] = pd.to_numeric(df["B11001_001M"], errors="coerce")
df["HH_CV"] = (df["HH_MOE"] / 1.645) / df["HH"]
df["HH_reliable"] = np.where(df["HH_CV"] <= 0.15, df["HH"], df["HH"] - df["HH_MOE"])


df["POP"] = pd.to_numeric(df["B01003_001E"], errors="coerce")
df["POP_MOE"] = pd.to_numeric(df["B01003_001M"], errors="coerce")
df["POP_CV"] = (df["POP_MOE"] / 1.645) / df["POP"]
df["POP_reliable"] = np.where(df["POP_CV"] <= 0.15, df["POP"], df["POP"] - df["POP_MOE"])

df["UNDERGRAD"] = pd.to_numeric(df["B14001_008E"], errors="coerce")
df["UNDERGRAD_MOE"] = pd.to_numeric(df["B14001_008M"], errors="coerce")
df["UNDERGRAD_CV"] = (df["UNDERGRAD_MOE"] / 1.645) / df["UNDERGRAD"]
df["UNDERGRAD_reliable"] = np.where(df["UNDERGRAD_CV"] <= 0.15, df["UNDERGRAD"], df["UNDERGRAD"] - df["UNDERGRAD_MOE"])


df["GRAD"] = pd.to_numeric(df["B14001_009E"], errors="coerce")
df["GRAD_MOE"] = pd.to_numeric(df["B14001_009M"], errors="coerce")
df["GRAD_CV"] = (df["GRAD_MOE"] / 1.645) / df["GRAD"]
df["GRAD_reliable"] = np.where(df["GRAD_CV"] <= 0.15, df["GRAD"], df["GRAD"] - df["GRAD_MOE"])



# Ensure no negative values in reliable columns
reliable_cols = ["HH_reliable", "POP_reliable", "UNDERGRAD_reliable", "GRAD_reliable"]
for col in reliable_cols:
    df[col] = df[col].clip(lower=0)


df = df[["TRACT_GEOID", "HH", "HH_MOE", "HH_reliable", "POP", "POP_MOE", "POP_reliable", "UNDERGRAD", "UNDERGRAD_MOE", "UNDERGRAD_reliable", "GRAD", "GRAD_MOE", "GRAD_reliable"]]


# write output to .csv
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {OUTPUT_CSV} with {len(df)} rows")
