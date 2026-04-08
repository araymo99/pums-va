import numpy as np
import pandas as pd

# access household and person-level PUMS data for Virginia (2024 5-year) - change file paths if needed
housing_path = "data/psam_h51.csv"   
person_path  = "data/psam_p51.csv"   

housing = pd.read_csv(housing_path)
persons = pd.read_csv(person_path)

# check that the number of rows in each file matches expectations 
print("Housing rows:", housing.shape)  
print("Person rows:", persons.shape)

# create table of householders only

# RELSHIP == 20 identifies the householder
householders = persons.loc[persons["RELSHIPP"] == 20].copy()

print("Householders:", householders.shape)

# keep only relevant person variables in the householders table

person_vars = [
    "SERIALNO",
    "AGEP",
    "SEX",
    "RAC1P",
    "HISP",
    "SCHL",
    "ESR",
    "PINCP",
    "PWGTP",
    "SCHG"
]

householders = householders[person_vars]

# merge householder file with housing units file

pums_hh = housing.merge(
    householders,
    on="SERIALNO",
    how="left",
    validate="one_to_one"
)

# adjust income and costs for inflation using the provided adjustment factors in the PUMS data dictionary (ADJINC for income, ADJHSG for housing costs)
# create adjustment factors 
pums_hh["ADJINC_FACTOR"] = pums_hh["ADJINC"] / 1_000_000
pums_hh["ADJHSG_FACTOR"] = pums_hh["ADJHSG"] / 1_000_000

#adjust income and housing cost variables by multiplying by the adjustment factors
pums_hh["FINCP_ADJ"] = pums_hh["FINCP"] * pums_hh["ADJINC_FACTOR"] #adjusted family income
pums_hh["HINCP_ADJ"] = pums_hh["HINCP"] * pums_hh["ADJINC_FACTOR"] #adjusted household income

pums_hh["ELEP_ADJ"]  = pums_hh["ELEP"]  * pums_hh["ADJHSG_FACTOR"] #monthly electricity
pums_hh["FULP_ADJ"]  = pums_hh["FULP"]  * pums_hh["ADJHSG_FACTOR"] #ANNUAL fuel
pums_hh["GASP_ADJ"]  = pums_hh["GASP"]  * pums_hh["ADJHSG_FACTOR"] #monthly gas
pums_hh["GRNTP_ADJ"] = pums_hh["GRNTP"] * pums_hh["ADJHSG_FACTOR"] #monthly gross rent
pums_hh["SMOCP_ADJ"] = pums_hh["SMOCP"] * pums_hh["ADJHSG_FACTOR"] #monthly owner cost

#total monthly adjusted energy cost
pums_hh["FUEL_MO_ADJ"] = pums_hh["FULP_ADJ"] / 12

pums_hh["ENERGY_MO_ADJ"] = (
    pums_hh["ELEP_ADJ"].fillna(0) +
    pums_hh["GASP_ADJ"].fillna(0) +
    pums_hh["FUEL_MO_ADJ"].fillna(0)
)


# create flags for undergraduate and graduate student householders based on SCHG variable (15 for undergrad, 16 for grad)
pums_hh["UNDERGRAD_HH"] = pums_hh["SCHG"] == 15
pums_hh["GRAD_HH"] = pums_hh["SCHG"] == 16


# create flags for housing cost burdened households (those with housing costs > 30% of income)
pums_hh["RENT_BURDENED"] = pums_hh["GRNTP_ADJ"] / pums_hh["HINCP_ADJ"] > 0.3
pums_hh["OWN_BURDENED"] = pums_hh["SMOCP_ADJ"] / pums_hh["HINCP_ADJ"] > 0.3
pums_hh["ANY_BURDENED"] = pums_hh["RENT_BURDENED"] | pums_hh["OWN_BURDENED"]    

# Replicate weights

REPLICATE_WEIGHTS = [f"WGTP{i}" for i in range(1, 81)]

def weighted_count_and_se(df, condition):
    """
    Returns (estimate, standard_error) for a weighted count
    using ACS replicate weights.
    """
    # Point estimate
    est = df.loc[condition, "WGTP"].sum()

    # Replicate estimates
    rep_ests = np.array([
        df.loc[condition, w].sum()
        for w in REPLICATE_WEIGHTS
    ])

    se = np.sqrt((4 / 80) * np.sum((rep_ests - est) ** 2))
    return est, se

def puma_student_hh_table(df):
    """
    For each PUMA, returns weighted counts and SE/CV for:
      - total households
      - undergraduate-led households (SCHG == 15)
      - graduate-student-led households (SCHG == 16)
    """
    records = []

    for puma, grp in df.groupby("PUMA"):
        all_hh      = pd.Series([True]  * len(grp), index=grp.index)
        undergrad   = grp["UNDERGRAD_HH"]
        grad        = grp["GRAD_HH"]

        est_total,  se_total  = weighted_count_and_se(grp, all_hh)
        est_ug,     se_ug     = weighted_count_and_se(grp, undergrad)
        est_grad,   se_grad   = weighted_count_and_se(grp, grad)

        records.append({
            "PUMA": puma,

            "total_hh_est":  est_total,
            "total_hh_se":   se_total,
            "total_hh_cv":   se_total  / est_total  if est_total  > 0 else np.nan,

            "undergrad_hh_est": est_ug,
            "undergrad_hh_se":  se_ug,
            "undergrad_hh_cv":  se_ug  / est_ug  if est_ug  > 0 else np.nan,

            "grad_hh_est":   est_grad,
            "grad_hh_se":    se_grad,
            "grad_hh_cv":    se_grad  / est_grad  if est_grad  > 0 else np.nan,
        })

    return pd.DataFrame(records).set_index("PUMA")


puma_table = puma_student_hh_table(pums_hh)
#print(puma_table)
#puma_table.to_csv("outputs/puma_student_led_hh.csv")

# create table of student-led households by PUMA and any housing cost burden

def puma_student_hh_burden_table(df):
    records = []

    for puma, grp in df.groupby("PUMA"):
        undergrad   = grp["UNDERGRAD_HH"]
        grad        = grp["GRAD_HH"]

        est_ug_burdened, se_ug_burdened = weighted_count_and_se(grp, undergrad & grp["ANY_BURDENED"])
        est_grad_burdened, se_grad_burdened = weighted_count_and_se(grp, grad & grp["ANY_BURDENED"])

        records.append({
            "PUMA": puma,

            "undergrad_hh_burdened_est": est_ug_burdened,
            "undergrad_hh_burdened_se":  se_ug_burdened,
            "undergrad_hh_burdened_cv":  se_ug_burdened  / est_ug_burdened  if est_ug_burdened  > 0 else np.nan,

            "grad_hh_burdened_est":   est_grad_burdened,
            "grad_hh_burdened_se":    se_grad_burdened,
            "grad_hh_burdened_cv":    se_grad_burdened  / est_grad_burdened  if est_grad_burdened  > 0 else np.nan,
        })

    return pd.DataFrame(records).set_index("PUMA") 


# calculate median income by student-led household status and PUMA using replicate weights
def weighted_median_income(df, condition):
    """
    Returns (estimate, standard_error) for a weighted median income
    using ACS replicate weights.
    """
    # account for negative incomes by filtering to positive incomes only for the median calculation
    df = df[df["HINCP_ADJ"] > 0]

    # account for negative weights by filtering to positive weights only for the median calculation
    for w in REPLICATE_WEIGHTS:
        df = df[df[w] > 0]

    # Point estimate
    est = df.loc[condition, "HINCP_ADJ"].median()

    # Replicate estimates
    rep_ests = np.array([
        df.loc[condition, "HINCP_ADJ"].sample(frac=1, weights=df[w], replace=True).median()
        for w in REPLICATE_WEIGHTS
    ])

    se = np.sqrt((4 / 80) * np.sum((rep_ests - est) ** 2))
    return est, se

def puma_student_hh_income_table(df):
    records = []

    for puma, grp in df.groupby("PUMA"):
        undergrad   = grp["UNDERGRAD_HH"]
        grad        = grp["GRAD_HH"]

        est_ug_income, se_ug_income = weighted_median_income(grp, undergrad)
        est_grad_income, se_grad_income = weighted_median_income(grp, grad)
        est_median_income, se_median_income = weighted_median_income(grp, pd.Series([True] * len(grp), index=grp.index))

        records.append({
            "PUMA": puma,

            "undergrad_hh_median_income_est": est_ug_income,
            "undergrad_hh_median_income_se":  se_ug_income,
            "undergrad_hh_median_income_cv":  se_ug_income  / est_ug_income  if est_ug_income  > 0 else np.nan,

            "grad_hh_median_income_est":   est_grad_income,
            "grad_hh_median_income_se":    se_grad_income,
            "grad_hh_median_income_cv":    se_grad_income  / est_grad_income  if est_grad_income  > 0 else np.nan,

            "median_income_est":   est_median_income,
            "median_income_se":    se_median_income,
            "median_income_cv":    se_median_income  / est_median_income  if est_median_income  > 0 else np.nan,
        })

    return pd.DataFrame(records).set_index("PUMA")

puma_income_table = puma_student_hh_income_table(pums_hh)
#print(puma_income_table)

# add median income to the burden table
puma_burden_table = puma_student_hh_burden_table(pums_hh)
puma_burden_table = puma_burden_table.join(puma_income_table)
#print(puma_burden_table)

puma_burden_table.to_csv("outputs/puma_student_cost_burden.csv")

# create a table of median household size for all households, undergraduate-led households, and graduate-led households by PUMA
def weighted_median_hh_size(df, condition):
    """
    Returns (estimate, standard_error) for a weighted median household size
    using ACS replicate weights.
    """

    # exclude negative estimates and weights for the median calculation
    df = df[df["NP"] > 0]
    for w in REPLICATE_WEIGHTS:
        df = df[df[w] > 0]

    # Point estimate
    est = df.loc[condition, "NP"] .median()

    # Replicate estimates
    rep_ests = np.array([
        df.loc[condition, "NP"].sample(frac=1, weights=df[w], replace=True).median()
        for w in REPLICATE_WEIGHTS
    ])

    se = np.sqrt((4 / 80) * np.sum((rep_ests - est) ** 2))
    return est, se

# calculate the weighted average household size for undergraduate-led households, graduate-led households, and all households by PUMA
def average_hh_size(df, condition):
    """
    Returns (estimate, standard_error) for a weighted average household size
    using ACS replicate weights.
    """ 
    #exclude negative weights for the average calculation
    for w in REPLICATE_WEIGHTS:
        df = df[df[w] > 0]

    # Point estimate
    est = np.average(df.loc[condition, "NP"], weights=df.loc[condition, "WGTP"])

    # Replicate estimates
    rep_ests = np.array([
        np.average(df.loc[condition, "NP"], weights=df.loc[condition, w])
        for w in REPLICATE_WEIGHTS
    ])

    se = np.sqrt((4 / 80) * np.sum((rep_ests - est) ** 2))
    return est, se

def puma_student_hh_size_table(df):
    records = []

    for puma, grp in df.groupby("PUMA"):
        undergrad   = grp["UNDERGRAD_HH"]
        grad        = grp["GRAD_HH"]

        est_ug_size, se_ug_size = weighted_median_hh_size(grp, undergrad)
        est_grad_size, se_grad_size = weighted_median_hh_size(grp, grad)
        est_median_size, se_median_size = weighted_median_hh_size(grp, pd.Series([True] * len(grp), index=grp.index))

        est_avg_ug_size, se_avg_ug_size = average_hh_size(grp, undergrad)
        est_avg_grad_size, se_avg_grad_size = average_hh_size(grp, grad)
        est_avg_median_size, se_avg_median_size = average_hh_size(grp, pd.Series([True] * len(grp), index=grp.index))

        records.append({
            "PUMA": puma,

            "undergrad_hh_median_size_est": est_ug_size,
            "undergrad_hh_median_size_se":  se_ug_size,
      

            "grad_hh_median_size_est":   est_grad_size,
            "grad_hh_median_size_se":    se_grad_size,
     

            "median_hh_size_est":   est_median_size,
            "median_hh_size_se":    se_median_size,

            "undergrad_hh_avg_size_est": est_avg_ug_size,
            "undergrad_hh_avg_size_se":  se_avg_ug_size,

            "grad_hh_avg_size_est":   est_avg_grad_size,
            "grad_hh_avg_size_se":    se_avg_grad_size,

            "median_hh_avg_size_est":   est_avg_median_size,
            "median_hh_avg_size_se":    se_avg_median_size,
     
        })

    return pd.DataFrame(records).set_index("PUMA")
puma_size_table = puma_student_hh_size_table(pums_hh)
puma_size_table.to_csv("outputs/puma_student_hh_size.csv")

# create a table of student households by renter or owner status and PUMA
pums_hh["RENTER_HH"] = pums_hh["TEN"] == 3
# tenure == 1 or 2 for homeowners
pums_hh["OWNER_HH"] = pums_hh["TEN"].isin([1, 2])


def puma_student_hh_tenure_table(df):
    records = []

    for puma, grp in df.groupby("PUMA"):
        undergrad   = grp["UNDERGRAD_HH"]
        grad        = grp["GRAD_HH"]

        est_ug_renter, se_ug_renter = weighted_count_and_se(grp, undergrad & grp["RENTER_HH"])
        est_ug_owner, se_ug_owner = weighted_count_and_se(grp, undergrad & grp["OWNER_HH"])

        est_grad_renter, se_grad_renter = weighted_count_and_se(grp, grad & grp["RENTER_HH"])
        est_grad_owner, se_grad_owner = weighted_count_and_se(grp, grad & grp["OWNER_HH"])

        records.append({
            "PUMA": puma,

            "undergrad_hh_renter_est": est_ug_renter,
            "undergrad_hh_renter_se":  se_ug_renter,

            "undergrad_hh_owner_est": est_ug_owner,
            "undergrad_hh_owner_se":  se_ug_owner,

            "grad_hh_renter_est": est_grad_renter,
            "grad_hh_renter_se":  se_grad_renter,

            "grad_hh_owner_est": est_grad_owner,
            "grad_hh_owner_se":  se_grad_owner,
        })

    return pd.DataFrame(records).set_index("PUMA")  
puma_tenure_table = puma_student_hh_tenure_table(pums_hh)
puma_tenure_table.to_csv("outputs/puma_student_hh_tenure.csv")

