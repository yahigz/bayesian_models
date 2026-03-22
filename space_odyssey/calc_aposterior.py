from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.readwrite import BIFReader


bn = BIFReader('spaceship.bif').get_model()
infer = VariableElimination(bn)

cases = [
    {'O2_Level': 'low', 'Temperature': 'cold', 'Alert_System': 'warning', 'Diagnosis': 'nominal'},
    {'O2_Level': 'low', 'Alert_System': 'warning', 'Diagnosis': 'anomaly'},
    {'Temperature': 'hot', 'Alert_System': 'silent', 'Diagnosis': 'nominal'},
    {'CO2_Level': 'high', 'Alert_System': 'silent', 'Porthole': 'danger'},
    {'AI_Test': 'pass', 'Temperature': 'hot', 'O2_Level': 'low', 'Alert_System': 'silent', 'Diagnosis': 'nominal'}
]

target_variable = 'Crew_Status'

for i, case in enumerate(cases, 1):
    print(f'Case {i}:')
    print(f'Evidence: {case}')
    q = infer.query(variables=[target_variable], evidence=case)
    print(f'P({target_variable} | evidence) = {q}')
    print()

switches = ['HAL_Switch', 'Thermostat', 'O2_Generator', 'CO2_Scrubber', 'Manoeuvre', 'Decompression']
observables = ['AI_Test', 'Diagnosis', 'Temperature', 'O2_Level', 'CO2_Level', 'Porthole', 'Alert_System']


def get_state_mapping(model, variable):
    """Returns (index_to_state, state_to_index) for a variable."""
    states = model.get_cpds(variable).state_names[variable]
    index_to_state = {idx: state for idx, state in enumerate(states)}
    state_to_index = {state: idx for idx, state in enumerate(states)}
    return index_to_state, state_to_index


def find_best_switch(evidence):
    crew_status_mapping_to_state, crew_status_mapping_to_index = get_state_mapping(bn, 'Crew_Status')
    best_switch = None
    best_prob = infer.query(variables=['Crew_Status'], evidence=evidence).values[crew_status_mapping_to_index['safe']]

    for switch in switches:
        p_switch = infer.query(variables=[switch], evidence=evidence)
        p_switch_0 = p_switch.values[0]
        p_switch_1 = p_switch.values[1]
        mapping = get_state_mapping(bn, switch)[0]

        evidence_with_switch_0 = {**evidence, switch: mapping[0]}
        evidence_with_switch_1 = {**evidence, switch: mapping[1]}

        p_crew_status_given_switch_0 = infer.query(variables=['Crew_Status'], evidence=evidence_with_switch_0).values
        p_crew_status_given_switch_1 = infer.query(variables=['Crew_Status'], evidence=evidence_with_switch_1).values

        p_crew_status_expected = p_switch_0 * p_crew_status_given_switch_1[crew_status_mapping_to_index['safe']] + p_switch_1 * p_crew_status_given_switch_0[crew_status_mapping_to_index['safe']]
        if p_crew_status_expected > best_prob:
            best_prob = p_crew_status_expected
            best_switch = switch

    return best_switch, best_prob

def find_best_two_switches(evidence):
    crew_status_mapping_to_state, crew_status_mapping_to_index = get_state_mapping(bn, 'Crew_Status')
    best_switches = None
    best_prob = infer.query(variables=['Crew_Status'], evidence=evidence).values[crew_status_mapping_to_index['safe']]

    for switch1 in switches:
        for switch2 in switches:
            if switch1 >= switch2:
                continue

            p_switch1 = infer.query(variables=[switch1], evidence=evidence)
            p_switch2 = infer.query(variables=[switch2], evidence=evidence)

            mapping1 = get_state_mapping(bn, switch1)[0]
            mapping2 = get_state_mapping(bn, switch2)[0]

            p_crew_status_expected = 0
            for val1 in [0, 1]:
                for val2 in [0, 1]:
                    evidence_with_switches = {**evidence, switch1: mapping1[val1], switch2: mapping2[val2]}
                    evidence_rev_switches = {**evidence, switch1: mapping1[1-val1], switch2: mapping2[1-val2]}
                    p_crew_status_given_switches = infer.query(variables=['Crew_Status'], evidence=evidence_rev_switches).values
                    p_crew_status_expected += p_switch1.values[val1] * p_switch2.values[val2] * p_crew_status_given_switches[crew_status_mapping_to_index['safe']]

            if p_crew_status_expected > best_prob:
                best_prob = p_crew_status_expected
                best_switches = (switch1, switch2)

    return best_switches, best_prob


def p_safe(evidence):
    _, crew_status_mapping_to_index = get_state_mapping(bn, 'Crew_Status')
    return float(infer.query(variables=['Crew_Status'], evidence=evidence).values[crew_status_mapping_to_index['safe']])


def best_assignment_for_two_switches(evidence, switch1, switch2):
    mapping1 = get_state_mapping(bn, switch1)[0]
    mapping2 = get_state_mapping(bn, switch2)[0]

    best_pair = None
    best_prob = -1.0
    for val1 in mapping1.values():
        for val2 in mapping2.values():
            candidate_evidence = {**evidence, switch1: val1, switch2: val2}
            prob = p_safe(candidate_evidence)
            if prob > best_prob:
                best_prob = prob
                best_pair = (val1, val2)

    return best_pair, best_prob


def evaluate_observation_candidate(evidence, observe_var, baseline_two_switch_prob):
    posterior = infer.query(variables=[observe_var], evidence=evidence)
    state_to_index = {state: idx for idx, state in enumerate(posterior.state_names[observe_var])}

    expected_prob_after_observe = 0.0
    prob_of_improvement = 0.0
    outcomes = []

    for state in posterior.state_names[observe_var]:
        state_prob = float(posterior.values[state_to_index[state]])
        evidence_after_observe = {**evidence, observe_var: state}
        _, best_prob_after_observe = find_best_two_switches(evidence_after_observe)

        gain = best_prob_after_observe - baseline_two_switch_prob
        if gain > 1e-12:
            prob_of_improvement += state_prob

        expected_prob_after_observe += state_prob * best_prob_after_observe
        outcomes.append({
            'state': state,
            'state_prob': state_prob,
            'best_prob_after_observe': best_prob_after_observe,
            'gain': gain,
        })

    return {
        'observe_var': observe_var,
        'expected_prob_after_observe': expected_prob_after_observe,
        'expected_gain': expected_prob_after_observe - baseline_two_switch_prob,
        'prob_of_improvement': prob_of_improvement,
        'outcomes': outcomes,
    }


def find_best_observe_switch_switch(evidence):
    baseline_two_switches, baseline_two_switch_prob = find_best_two_switches(evidence)

    candidates = [var for var in observables if var not in evidence]
    if not candidates:
        return {
            'baseline_two_switches': baseline_two_switches,
            'baseline_two_switch_prob': baseline_two_switch_prob,
            'best_observe': None,
            'ranked': [],
        }

    ranked = [
        evaluate_observation_candidate(evidence, observe_var, baseline_two_switch_prob)
        for observe_var in candidates
    ]
    ranked.sort(key=lambda x: x['expected_prob_after_observe'], reverse=True)

    return {
        'baseline_two_switches': baseline_two_switches,
        'baseline_two_switch_prob': baseline_two_switch_prob,
        'best_observe': ranked[0],
        'ranked': ranked,
    }


def run_observe_switch_switch_interactive(evidence):
    report = find_best_observe_switch_switch(evidence)

    print('\n=== observe -> switch -> switch ===')
    print(f'Current evidence: {evidence}')
    print(
        f"Baseline (without new observe), best two-switch expected P(Crew_Status=safe): "
        f"{report['baseline_two_switch_prob']:.6f} with {report['baseline_two_switches']}"
    )

    if not report['best_observe']:
        print('No available observable variables left for this evidence.')
        return

    print('\nObservation candidates (sorted):')
    for row in report['ranked']:
        print(
            f"- {row['observe_var']}: "
            f"E[P_safe after observe+2switch]={row['expected_prob_after_observe']:.6f}, "
            f"expected gain={row['expected_gain']:.6f}, "
            f"P(improve)={row['prob_of_improvement']:.6f}"
        )

    best = report['best_observe']
    observe_var = best['observe_var']
    allowed_states = list(get_state_mapping(bn, observe_var)[0].values())

    print(
        f"\nSuggested observe first: {observe_var} "
        f"(expected gain {best['expected_gain']:.6f}, P(improve)={best['prob_of_improvement']:.6f})"
    )
    print(f'Possible observed values: {allowed_states}')

    observed_value = input(f'Enter observed value for {observe_var}: ').strip()
    while observed_value not in allowed_states:
        print(f'Invalid value. Allowed: {allowed_states}')
        observed_value = input(f'Enter observed value for {observe_var}: ').strip()

    evidence_after_observe = {**evidence, observe_var: observed_value}
    best_two_switches, best_prob = find_best_two_switches(evidence_after_observe)

    if best_two_switches is None:
        print('No two-switch improvement found. Keep current plan.')
        return

    (switch1, switch2) = best_two_switches
    (value1, value2), assignment_prob = best_assignment_for_two_switches(
        evidence_after_observe,
        switch1,
        switch2,
    )

    print('\nRecommended actions after observation:')
    print(f'1) Set {switch1} -> {value1}')
    print(f'2) Set {switch2} -> {value2}')
    print(f'Expected P(Crew_Status=safe) after these actions: {assignment_prob:.6f}')
    print(f'(Two-switch planner estimate for this evidence: {best_prob:.6f})')


if __name__ == '__main__':
    print('\nChoose scenario index (1..5) for interactive observe->switch->switch planning.')
    selected = input('Scenario index: ').strip()
    while selected not in {'1', '2', '3', '4', '5'}:
        selected = input('Please enter a number from 1 to 5: ').strip()

    run_observe_switch_switch_interactive(cases[int(selected) - 1])
