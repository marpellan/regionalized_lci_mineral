import bw2analyzer as ba
import bw2calc as bc
import bw2data as bd
import bw2io as bi
import brightway2 as bw
import pandas as pd
import numpy as np
import datetime
import math # for pedigree matrix


### LCI ###
def get_inventory_dataset(inventories, database_names):
    """
    Function to find the dataset in the specified databases.
    """
    inventory_ds = {}
    for rm_name, (activity_name, ref_product, location) in inventories.items():
        match_found = False

        # Iterate over the list of database names
        for database_name in database_names:
            db = bw.Database(database_name)
            matches = [ds for ds in db if ds["name"] == activity_name
                       and ds["reference product"] == ref_product
                       and ds["location"] == location]

            if matches:
                inventory_ds[rm_name] = matches[0]
                match_found = True
                break  # Stop searching once a match is found

        if not match_found:
            print(f"No match found for {rm_name} in provided databases")
    return inventory_ds


def create_pedigree_matrix(pedigree_scores: tuple, exc_amount: float):
    """
    Function from Istrate et al (2024)

    This function returns a dict containing the pedigree matrix dict and loc and scale values
    that can be used to update exchanges in a dataset dict

    The pedigree matrix dictionary is created using the scores provided in the LCI Excel file.

    The code to calcualte the loc and scale values is based on https://github.com/brightway-lca/pedigree_matrix,
    which is published by Chris Mutel under an BSD 3-Clause License (2021).

    :param pedigree_scores: tuple of pedigree scores
    :param exc_amount: exchange amount
    :return dict:
    """

    VERSION_2 = {
        "reliability": (1.0, 1.54, 1.61, 1.69, 1.69),
        "completeness": (1.0, 1.03, 1.04, 1.08, 1.08),
        "temporal correlation": (1.0, 1.03, 1.1, 1.19, 1.29),
        "geographical correlation": (1.0, 1.04, 1.08, 1.11, 1.11),
        "further technological correlation": (1.0, 1.18, 1.65, 2.08, 2.8),
        "sample size": (1.0, 1.0, 1.0, 1.0, 1.0),
    }

    pedigree_scores_dict = {
        'reliability': pedigree_scores[0],
        'completeness': pedigree_scores[1],
        'temporal correlation': pedigree_scores[2],
        'geographical correlation': pedigree_scores[3],
        'further technological correlation': pedigree_scores[4]
    }

    assert len(pedigree_scores) in (5, 6), "Must provide either 5 or 6 factors"
    if len(pedigree_scores) == 5:
        pedigree_scores = pedigree_scores + (1,)

    factors = [VERSION_2[key][index - 1] for key, index in pedigree_scores_dict.items()]

    basic_uncertainty: float = 1.0
    values = [basic_uncertainty] + factors

    scale = math.sqrt(sum([math.log(x) ** 2 for x in values])) / 2
    loc = math.log(abs(exc_amount))

    pedigree_dict = {
        'uncertainty type': 2,
        'loc': loc,
        'scale': scale,
        "pedigree": pedigree_scores_dict,
    }
    return pedigree_dict


### LCA calculations ###
def init_simple_lca(activity):
    """
    Initialize simple LCA object
    """
    lca = bw.LCA({activity.key: 1})
    lca.lci()
    #lca.lcia()
    #lca.score

    return lca


def multi_lcia(lca, activity, lcia_methods, amount=1):
    """
    Calculate multiple impact categories, including units in the result keys.

    Parameters:
    - lca: lca object
    - activity (object): An activity object representing the product or process being assessed.
    - lcia_methods (dict): A dictionary of impact categories and their corresponding method.
                           The keys are the names of the impact categories and the values are the methods to be used for each category.
    - amount (float): The functional unit of the assessment. Defaults to 1.

    Returns:
    - multi_lcia_results (dict): A dictionary with keys as impact categories including units and values as scores.
                                 Each key includes the unit in the format `impact category (unit)`.
    """

    results = {}
    lca.redo_lci({activity.key: amount})

    for impact_name, method in lcia_methods.items():
        lca.switch_method(method)
        lca.lcia()

        # Retrieve the unit of the method
        method_obj = bw.Method(method)
        unit = method_obj.metadata.get("unit", "Unknown unit")

        # Add unit to impact name in the dictionary key
        results[f"{impact_name} ({unit})"] = lca.score

    return results


# Function for contribution analysis with customizable threshold and percentage calculation
def multi_contribution_analysis(lca, lcia_methods, top_n=10, threshold=0.01):
    """
    Perform contribution analysis on multiple LCIA methods with percentage impact and filtering.

    Parameters:
    - lca: Brightway LCA object.
    - lcia_methods: Dictionary of LCIA methods with method names as keys.
    - top_n: Number of top processes to include in the contribution analysis.
    - threshold: Minimum percentage contribution to include in the results.

    Returns:
    - results: Dictionary of contribution analysis results for each LCIA method.
    """
    results = {}

    for impact_name, method in lcia_methods.items():
        # Calculate the total impact score for this method
        lca.switch_method(method)
        lca.lcia()
        total_score = lca.score

        # Perform contribution analysis for top `top_n` processes
        contributions = ba.ContributionAnalysis().annotated_top_processes(lca, limit=top_n)

        # Store contributions as a list of dictionaries with percentage and reference product
        results[impact_name] = [
            {
                "score": score,
                "quantity": quantity,
                "percentage": (score / total_score) * 100,  # Calculate percentage
                "name": process["name"],
                "reference product": process.get("reference product", "")
            }
            for score, quantity, process in contributions if (score / total_score) * 100 >= threshold * 100
            # Apply threshold
        ]

    return results


### Scale up ###
# Calculate projected impacts using the mapping
def calculate_projected_impacts(production_df, impact_df, mapping):
    projections = []

    for mineral in production_df['Mineral'].unique():
        # Use the mapping dictionary to get the corresponding raw material
        raw_material = mapping.get(mineral)

        if raw_material:
            # Fetch impact factors for the mapped raw material
            material_impacts = impact_df[impact_df['Mineral'] == raw_material]

            if not material_impacts.empty:
                impacts_per_kt = material_impacts.iloc[0, 1:].to_dict()  # Extract impact per kilotonne as a dict

                # Filter production data for the mineral
                mineral_data = production_df[production_df['Mineral'] == mineral]

                for _, row in mineral_data.iterrows():
                    year = row['Year']
                    production_kilotons = row['Value']

                    # Calculate impacts for each category
                    annual_impacts = {f"{category}": production_kilotons * impact_per_kt * 1000000
                                      for category, impact_per_kt in impacts_per_kt.items()}
                    annual_impacts['Year'] = year
                    annual_impacts['Mineral'] = mineral
                    projections.append(annual_impacts)

    # Convert list of dictionaries to DataFrame
    projected_impacts_df = pd.DataFrame(projections)

    return projected_impacts_df


### Ore grade ###
def ore_grade_decline(t, a, b):
    """
    Calculate the ore grade at a given time based on a power regression model.

    Parameters:
    - t (float or np.array): Time in years (e.g., since a baseline year).
    - a (float): Constant for the initial ore grade level.
    - b (float): Exponent that defines the rate of ore grade decline.

    Returns:
    - float or np.array: Estimated ore grade at time t.
    """
    return a * (t ** b)


def energy_ore_grade(ore_grade, r, q):
    """
    Calculate the energy requirement based on ore grade.

    Parameters:
    - ore_grade (float or np.array): The ore grade value(s), e.g., in percentage or fraction.
    - r (float): Constant for energy-ore grade relation, specific to the metal.
    - q (float): Exponent that defines sensitivity of energy to ore grade changes.

    Returns:
    - float or np.array: Energy requirement (MJ/kg) based on ore grade.
    """
    return r * np.power(ore_grade, q)


def future_energy_requirement(t, c, d):
    """
    Calculate the energy requirement over time as ore grades decline.

    Parameters:
    - t (float or np.array): Time in years (e.g., years since baseline).
    - c (float): Constant for the initial energy requirement.
    - d (float): Exponent that defines the growth of energy requirements over time.

    Returns:
    - float or np.array: Projected energy requirement (MJ/kg) at time t.
    """
    return c * (t ** d)


def percentage_change_energy(E_t, E_0):
    """
    Calculate the percentage change in energy requirements over time.

    Parameters:
    - E_t (float or np.array): Future energy requirement (MJ/kg) at time t.
    - E_0 (float): Baseline energy requirement (MJ/kg) at a reference year.

    Returns:
    - float or np.array: Percentage increase in energy requirements relative to baseline.
    """
    return (E_t - E_0) / E_0


def modeling_factor(E_0, E_t):
    """
    Calculate the modeling factor to scale energy requirements in LCA models.

    Parameters:
    - E_0 (float): Baseline energy requirement (MJ/kg) at the reference year.
    - E_t (float or np.array): Future energy requirement (MJ/kg) at time t.

    Returns:
    - float or np.array: Modeling factor for scaling energy in LCA.
    """
    return E_0 / E_t