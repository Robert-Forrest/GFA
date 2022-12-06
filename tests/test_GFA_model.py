import cerebral as cb


def test_GFA_model():

    cb.setup(
        {
            "targets": [
                {"name": "Tl", "loss": "Huber", "weight": 0.1},
                {"name": "Tg", "loss": "Huber", "weight": 0.1},
                {"name": "Tx", "loss": "Huber", "weight": 0.1},
                {"name": "GFA", "type": "categorical", "weight": 10},
                {"name": "Dmax", "loss": "Huber", "weight": 100},
            ],
            "train": {"max_epochs": 1000, "dropout": 0.0, "max_norm": 10},
            "input_features": [
                "atomic_number",
                "periodic_number",
                "mass",
                "group",
                "radius",
                "atomic_volume",
                "period",
                "protons",
                "neutrons",
                "electrons",
                "valence_electrons",
                "valence",
                "electron_affinity",
                "ionisation_energies",
                "wigner_seitz_electron_density",
                "work_function",
                "mendeleev_universal_sequence",
                "chemical_scale",
                "mendeleev_pettifor",
                "mendeleev_modified",
                "electronegativity_pauling",
                "electronegativity_miedema",
                "electronegativity_mulliken",
                "melting_temperature",
                "boiling_temperature",
                "fusion_enthalpy",
                "vaporisation_enthalpy",
                "molar_heat_capacity",
                "thermal_conductivity",
                "thermal_expansion",
                "density",
                "cohesive_energy",
                "debye_temperature",
                "chemical_hardness",
                "chemical_potential",
                "theoretical_density",
                "atomic_volume_deviation",
                "s_valence",
                "p_valence",
                "d_valence",
                "f_valence",
                "structure_deviation",
                "ideal_entropy",
                "ideal_entropy_xia",
                "mismatch_entropy",
                "mixing_entropy",
                "mixing_enthalpy",
                "mixing_Gibbs_free_energy",
                "block_deviation",
                "series_deviation",
                "viscosity",
                "lattice_distortion",
                "shell_valence_electron_concentration_ratio",
                "shell_mendeleev_number_ratio",
                "mismatch_PHS",
                "mixing_PHS",
                "mixing_PHSS",
            ],
            "data": ["data/data.csv"],
            "plot": {"model": False, "data": False},
        }
    )

    data = cb.features.load_data()

    model, history, train_ds = cb.models.train_model(data, max_epochs=1000)

    (
        train_eval,
        metrics,
    ) = cb.models.evaluate_model(model, train_ds)

    print(metrics)
    assert metrics["Tl"]["train"]["R_sq"] > 0.5
    assert metrics["Tg"]["train"]["R_sq"] > 0.5
    assert metrics["Tx"]["train"]["R_sq"] > 0.5
    assert metrics["GFA"]["train"]["accuracy"] > 0.5


test_GFA_model()
