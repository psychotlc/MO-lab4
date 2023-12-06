from multicriteria import Multicriteria

import json

if __name__ == "__main__":

    with open("input.json", "r") as input_file:
        mc_instance = Multicriteria(json.load(input_file))

        mc_instance.out_weight()

        mc_instance.main_criteria_method()

        mc_instance.pareto_method()

        mc_instance.weight_and_combined_method()

        mc_instance.hierarchies_analysis_method()