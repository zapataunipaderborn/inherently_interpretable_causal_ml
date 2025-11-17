import numpy as np
import pandas as pd

def simulation(
    n_samples=4000,
    seed=42,
    intervention=None
):
    np.random.seed(seed)
    
    # Generate a pattern
    t = np.linspace(0.1, 4 * np.pi, n_samples)
    data = pd.DataFrame()

    # "size" — base exogenous variable
    data["size"] = (
        250
        + 120 * np.sin(t)
        + 80 * np.sin(1.5 * t)
        + 30 * (t ** 2) / 100
        + np.random.normal(0, 10, n_samples)
    )
    # If there is an intervention set the size to the intervention value and compute the rest of the simulation
    if intervention is not None:
        data["size"] = intervention

    # "material" — exogenous with its own independent pattern
    data["material"] = (
        3
        + 2.5 * np.log1p(t)
        + 10 * np.sin(t / 2)
        + np.random.normal(0, 1, n_samples)
    )

    # "productivity" — depends on size and material
    data["productivity"] = (
        50
        + 0.6 * data["size"]
        + 5 * np.log1p(data["material"])
        + 0.3 * data["size"] * np.log1p(np.abs(data["material"]))
        + np.random.normal(0, 5, n_samples)
    )

    # "energy" — depends on productivity and size
    data["energy"] = (
        1e4
        + (data["size"] ** 2)
        + 130 * data["productivity"]
        + np.random.normal(0, 200, n_samples)
    )
    data["energy"] = np.clip(data["energy"], 0, None)

    # "personnel" — depends only on size
    data["personnel"] = (
        15
        + 0.001 * (data["size"] ** 2)
        + np.random.normal(0, 1, n_samples)
    )

    # Cleanup
    data = data.dropna().reset_index(drop=True)
    return data