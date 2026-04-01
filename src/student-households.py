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
pums_hh["FINCP_ADJ"] = pums_hh["FINCP"] * pums_hh["ADJINC_FACTOR"]
pums_hh["HINCP_ADJ"] = pums_hh["HINCP"] * pums_hh["ADJINC_FACTOR"]

pums_hh["ELEP_ADJ"]  = pums_hh["ELEP"]  * pums_hh["ADJHSG_FACTOR"] #monght electirc
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
print(puma_table)
puma_table.to_csv("outputs/puma_student_led_hh.csv")