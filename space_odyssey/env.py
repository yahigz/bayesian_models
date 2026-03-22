from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader
import logging
import random
import networkx as nx


logging.getLogger('pgmpy').setLevel(logging.ERROR)


OBS_VARS = {
    'AI_Test', 'Diagnosis', 'Temperature', 'O2_Level',
    'CO2_Level', 'Porthole', 'Alert_System'
}
DO_VARS =  {
    'HAL_Switch', 'Thermostat', 'O2_Generator',
    'CO2_Scrubber', 'Manoeuvre', 'Decompression'
}
HIDDEN_VARS = {'HAL', 'System_Age', 'Life_Support', 'Meteor_Shower', 'Alien_Attack'}
TARGET_VARS = {'Crew_Status'}
ACTION_POINTS = 5


class SpaceOdysseySimulator:
    model: DiscreteBayesianNetwork
    def __init__(
            self,
            model: DiscreteBayesianNetwork,
            initial_state: dict = None,
            seed: int = None,
    ):
        self.seed = seed
        self._rng = random.Random(seed)
        # define parameters
        self.model = model
        self.obs_vars = OBS_VARS
        self.do_vars = DO_VARS
        self.hidden_vars = HIDDEN_VARS
        self.target_vars = TARGET_VARS
        self.start_action_points = ACTION_POINTS
        self.infer = VariableElimination(self.model)

        self.steps = 0
        # track available action points
        self.action_points = self.start_action_points
        self.known_evidence = dict(initial_state) if initial_state else {}
        # initialize state
        self.initial_state = initial_state
        self._state = self.model.simulate(
            n_samples=1,
            evidence=self.initial_state,
            seed=self._next_seed(),
        ).iloc[0].to_dict()

    def _next_seed(self):
        return self._rng.randint(0, 2**32 - 1)

    def reset(self):
        """
        Reset simulation state
        :return:
        """
        self.steps = 0
        self.action_points = self.start_action_points
        self.known_evidence = dict(self.initial_state) if self.initial_state else {}
        self._rng = random.Random(self.seed)
        self._state = self.model.simulate(
            n_samples=1,
            evidence=self.initial_state,
            seed=self._next_seed(),
        ).iloc[0].to_dict()

    def observe(self, variable):
        """
        Reveal state value of a specific variable
        :param variable: name from obs_vars set
        :return: observation state
        """
        print(f'*** Step {self.steps} ***')
        if variable not in self.obs_vars:
            print(f"We can't observe {variable} directly.")
            return None

        if self.action_points < 1:
            print(f"You have no action points left.")
            return None

        obs_state = self._state[variable]
        self.action_points -= 1
        self.steps += 1
        self.known_evidence[variable] = obs_state

        print(f'You send your crew member to check {variable}.')
        print(f'{variable} is reported to be in {obs_state} state.')
        print(f'You have {self.action_points} action points left.')
        print()
        return obs_state

    def act(self, variable, value):
        """
        Force state of a variable (intervention)
        :param variable: name of a variable from do_vars set
        :param value: desirable state value
        :return:
        """
        print(f'*** Step {self.steps} ***')
        if variable not in self.do_vars:
            print(f"We can't change {variable} directly.")
            return

        if self.action_points < 2:
            print(f"Not enough action points.")
            return

        # apply do operator
        # first, we should remove all descendants of the variable from the state
        ds = nx.descendants(self.model, variable)
        self._state = {key: value for key, value in self._state.items() if key not in ds.union({variable})}
        # Old observed descendants become stale after intervention and resampling.
        for d in ds:
            self.known_evidence.pop(d, None)
        self.known_evidence[variable] = value
        # then resample state values using state as evidence
        self._state = self.model.simulate(
            n_samples=1,
            do={variable: value},
            evidence=self._state,
            seed=self._next_seed(),
        ).iloc[0].to_dict()
        self.action_points -= 2
        self.steps += 1

        print(f'You send your crew member to fix {variable}.')
        print(f'{variable} is now {value}.')
        print(f'You have {self.action_points} action points left.')
        print()

    def finish(self):
        """
        Reveal states of target variables
        :return:
        """
        print('*** Finishing Simulation ***')
        for tv in self.target_vars:
            print(f'{tv}: {self._state[tv]}')

    def _get_state_mapping(self, variable):
        states = self.model.get_cpds(variable).state_names[variable]
        index_to_state = {idx: state for idx, state in enumerate(states)}
        state_to_index = {state: idx for idx, state in enumerate(states)}
        return index_to_state, state_to_index

    def _p_safe(self, evidence):
        _, crew_status_mapping_to_index = self._get_state_mapping('Crew_Status')
        q = self.infer.query(variables=['Crew_Status'], evidence=evidence)
        return float(q.values[crew_status_mapping_to_index['safe']])

    def _inverse_state(self, variable, state):
        index_to_state, state_to_index = self._get_state_mapping(variable)
        if len(index_to_state) != 2:
            raise ValueError(f'Variable {variable} is not binary, cannot invert state.')
        idx = state_to_index[state]
        return index_to_state[1 - idx]

    def _apply_switch(self, variable):
        """Apply switch action as inversion of current true state in simulator."""
        current_state = self._state[variable]
        target_state = self._inverse_state(variable, current_state)
        self.act(variable, target_state)
        return target_state

    def _expected_p_safe_after_switch(self, evidence, switch):
        posterior = self.infer.query(variables=[switch], evidence=evidence)
        states = posterior.state_names[switch]
        state_to_index = {state: idx for idx, state in enumerate(states)}

        expected = 0.0
        for state in states:
            p_state = float(posterior.values[state_to_index[state]])
            switched_state = self._inverse_state(switch, state)
            switched_evidence = {**evidence, switch: switched_state}
            expected += p_state * self._p_safe(switched_evidence)
        return expected

    def _expected_p_safe_after_two_switches(self, evidence, switch1, switch2):
        posterior = self.infer.query(variables=[switch1, switch2], evidence=evidence)
        states1 = posterior.state_names[switch1]
        states2 = posterior.state_names[switch2]
        idx1 = {state: i for i, state in enumerate(states1)}
        idx2 = {state: i for i, state in enumerate(states2)}

        expected = 0.0
        for state1 in states1:
            for state2 in states2:
                p_joint = float(posterior.values[idx1[state1], idx2[state2]])
                switched_evidence = {
                    **evidence,
                    switch1: self._inverse_state(switch1, state1),
                    switch2: self._inverse_state(switch2, state2),
                }
                expected += p_joint * self._p_safe(switched_evidence)
        return expected

    def _find_best_two_switches(self, evidence):
        best_switches = None
        best_prob = self._p_safe(evidence)

        do_vars = sorted(list(self.do_vars))
        for i in range(len(do_vars)):
            for j in range(i + 1, len(do_vars)):
                switch1 = do_vars[i]
                switch2 = do_vars[j]
                pair_best_prob = self._expected_p_safe_after_two_switches(
                    evidence,
                    switch1,
                    switch2,
                )

                if pair_best_prob > best_prob + 1e-12:
                    best_prob = pair_best_prob
                    best_switches = (switch1, switch2)

        return best_switches, best_prob

    def _evaluate_observation_candidate(self, evidence, observe_var, baseline_two_switch_prob):
        posterior = self.infer.query(variables=[observe_var], evidence=evidence)
        state_to_index = {
            state: idx for idx, state in enumerate(posterior.state_names[observe_var])
        }

        expected_prob_after_observe = 0.0
        prob_of_improvement = 0.0

        for state in posterior.state_names[observe_var]:
            state_prob = float(posterior.values[state_to_index[state]])
            evidence_after_observe = {**evidence, observe_var: state}
            _, best_prob_after_observe = self._find_best_two_switches(evidence_after_observe)

            gain = best_prob_after_observe - baseline_two_switch_prob
            if gain > 1e-12:
                prob_of_improvement += state_prob

            expected_prob_after_observe += state_prob * best_prob_after_observe

        return {
            'observe_var': observe_var,
            'expected_prob_after_observe': expected_prob_after_observe,
            'expected_gain': expected_prob_after_observe - baseline_two_switch_prob,
            'prob_of_improvement': prob_of_improvement,
        }

    def _find_best_single_switch(self, evidence, excluded_switches=None):
        excluded_switches = excluded_switches or set()
        best_action = None
        best_prob = self._p_safe(evidence)

        for switch in sorted(self.do_vars - set(excluded_switches)):
            prob = self._expected_p_safe_after_switch(evidence, switch)
            if prob > best_prob + 1e-12:
                best_prob = prob
                best_action = switch

        return best_action, best_prob

    def _expected_best_single_after_observe(self, evidence, observe_var, excluded_switches=None):
        baseline_prob = self._find_best_single_switch(
            evidence, excluded_switches=excluded_switches
        )[1]
        posterior = self.infer.query(variables=[observe_var], evidence=evidence)
        state_to_index = {
            state: idx for idx, state in enumerate(posterior.state_names[observe_var])
        }

        expected_prob = 0.0
        prob_of_improvement = 0.0
        for state in posterior.state_names[observe_var]:
            state_prob = float(posterior.values[state_to_index[state]])
            candidate_evidence = {**evidence, observe_var: state}
            _, best_prob_after_observe = self._find_best_single_switch(
                candidate_evidence,
                excluded_switches=excluded_switches,
            )
            expected_prob += state_prob * best_prob_after_observe
            if best_prob_after_observe > baseline_prob + 1e-12:
                prob_of_improvement += state_prob

        return {
            'observe_var': observe_var,
            'expected_prob_after_observe': expected_prob,
            'expected_gain': expected_prob - baseline_prob,
            'prob_of_improvement': prob_of_improvement,
        }

    def choose_plan_switch_observe_switch(self):
        """Plan first switch and best follow-up observe for switch->observe->switch."""
        evidence = dict(self.known_evidence)
        baseline_prob = self._p_safe(evidence)
        available_observes = sorted(list(self.obs_vars - set(evidence.keys())))

        best_plan = None
        best_expected_prob = baseline_prob

        for first_switch in sorted(self.do_vars):
            posterior_first = self.infer.query(variables=[first_switch], evidence=evidence)
            first_states = posterior_first.state_names[first_switch]
            first_state_to_idx = {state: idx for idx, state in enumerate(first_states)}

            if not available_observes:
                expected_prob = 0.0
                for state in first_states:
                    p_state = float(posterior_first.values[first_state_to_idx[state]])
                    switched_evidence = {**evidence, first_switch: self._inverse_state(first_switch, state)}
                    _, branch_prob = self._find_best_single_switch(
                        switched_evidence,
                        excluded_switches={first_switch},
                    )
                    expected_prob += p_state * branch_prob

                if expected_prob > best_expected_prob + 1e-12:
                    best_expected_prob = expected_prob
                    best_plan = {
                        'first_switch': first_switch,
                        'observe_var': None,
                        'expected_prob': expected_prob,
                        'observe_details': None,
                    }
                continue

            best_observe_for_first = None
            for observe_var in available_observes:
                expected_prob = 0.0
                weighted_gain = 0.0
                weighted_improve = 0.0

                for state in first_states:
                    p_state = float(posterior_first.values[first_state_to_idx[state]])
                    switched_evidence = {**evidence, first_switch: self._inverse_state(first_switch, state)}
                    observe_report = self._expected_best_single_after_observe(
                        switched_evidence,
                        observe_var,
                        excluded_switches={first_switch},
                    )
                    expected_prob += p_state * observe_report['expected_prob_after_observe']
                    weighted_gain += p_state * observe_report['expected_gain']
                    weighted_improve += p_state * observe_report['prob_of_improvement']

                candidate = {
                    'observe_var': observe_var,
                    'expected_prob_after_observe': expected_prob,
                    'expected_gain': weighted_gain,
                    'prob_of_improvement': weighted_improve,
                }

                if (
                    best_observe_for_first is None
                    or candidate['expected_prob_after_observe']
                    > best_observe_for_first['expected_prob_after_observe'] + 1e-12
                ):
                    best_observe_for_first = candidate

            expected_prob = best_observe_for_first['expected_prob_after_observe']
            if expected_prob > best_expected_prob + 1e-12:
                best_expected_prob = expected_prob
                best_plan = {
                    'first_switch': first_switch,
                    'observe_var': best_observe_for_first['observe_var'],
                    'expected_prob': expected_prob,
                    'observe_details': best_observe_for_first,
                }

        return {
            'baseline_prob': baseline_prob,
            'best_plan': best_plan,
        }

    def _best_expected_prob_after_k_observes_then_switch(
        self,
        evidence,
        available_observes,
        remaining_observes,
        excluded_switches=None,
        cache=None,
    ):
        excluded_switches = excluded_switches or set()
        cache = cache if cache is not None else {}
        key = (
            tuple(sorted(evidence.items())),
            tuple(sorted(available_observes)),
            remaining_observes,
            tuple(sorted(excluded_switches)),
        )
        if key in cache:
            return cache[key]

        if remaining_observes == 0 or not available_observes:
            result = self._find_best_single_switch(
                evidence,
                excluded_switches=excluded_switches,
            )[1]
            cache[key] = result
            return result

        best_expected = -1.0
        for observe_var in available_observes:
            posterior = self.infer.query(variables=[observe_var], evidence=evidence)
            state_to_index = {
                state: idx for idx, state in enumerate(posterior.state_names[observe_var])
            }
            next_available = tuple(v for v in available_observes if v != observe_var)

            expected_prob = 0.0
            for state in posterior.state_names[observe_var]:
                state_prob = float(posterior.values[state_to_index[state]])
                next_evidence = {**evidence, observe_var: state}
                expected_prob += state_prob * self._best_expected_prob_after_k_observes_then_switch(
                    next_evidence,
                    next_available,
                    remaining_observes - 1,
                    excluded_switches=excluded_switches,
                    cache=cache,
                )

            if expected_prob > best_expected:
                best_expected = expected_prob

        cache[key] = best_expected
        return best_expected

    def _choose_best_observe_for_remaining_depth(
        self,
        evidence,
        available_observes,
        remaining_observes,
        excluded_switches=None,
    ):
        excluded_switches = excluded_switches or set()
        baseline = self._best_expected_prob_after_k_observes_then_switch(
            evidence,
            available_observes,
            remaining_observes - 1,
            excluded_switches=excluded_switches,
            cache={},
        )

        best = None
        for observe_var in available_observes:
            posterior = self.infer.query(variables=[observe_var], evidence=evidence)
            state_to_index = {
                state: idx for idx, state in enumerate(posterior.state_names[observe_var])
            }
            next_available = tuple(v for v in available_observes if v != observe_var)

            expected_prob = 0.0
            prob_of_improvement = 0.0
            for state in posterior.state_names[observe_var]:
                state_prob = float(posterior.values[state_to_index[state]])
                next_evidence = {**evidence, observe_var: state}
                value = self._best_expected_prob_after_k_observes_then_switch(
                    next_evidence,
                    next_available,
                    remaining_observes - 1,
                    excluded_switches=excluded_switches,
                    cache={},
                )
                expected_prob += state_prob * value
                if value > baseline + 1e-12:
                    prob_of_improvement += state_prob

            candidate = {
                'observe_var': observe_var,
                'expected_prob': expected_prob,
                'expected_gain': expected_prob - baseline,
                'prob_of_improvement': prob_of_improvement,
            }
            if best is None or candidate['expected_prob'] > best['expected_prob'] + 1e-12:
                best = candidate

        return best

    def choose_best_observation_for_oss(self):
        """Select best first observe for observe->switch->switch strategy."""
        evidence = dict(self.known_evidence)
        baseline_two_switches, baseline_two_switch_prob = self._find_best_two_switches(evidence)

        candidates = sorted(list(self.obs_vars - set(evidence.keys())))
        if not candidates:
            return {
                'baseline_two_switches': baseline_two_switches,
                'baseline_two_switch_prob': baseline_two_switch_prob,
                'best_observe': None,
                'ranked': [],
            }

        ranked = [
            self._evaluate_observation_candidate(evidence, observe_var, baseline_two_switch_prob)
            for observe_var in candidates
        ]
        ranked.sort(key=lambda x: x['expected_prob_after_observe'], reverse=True)
        return {
            'baseline_two_switches': baseline_two_switches,
            'baseline_two_switch_prob': baseline_two_switch_prob,
            'best_observe': ranked[0],
            'ranked': ranked,
        }

    def run_strategy_observe_switch_switch(self):
        """Autonomous policy: observe one variable, then apply two interventions."""
        report = self.choose_best_observation_for_oss()
        best = report['best_observe']
        if not best:
            print('No observable variables left for strategy execution.')
            return report

        print('=== Running Strategy: observe -> switch -> switch ===')
        print(
            f"Planned first observe: {best['observe_var']} "
            f"(expected gain={best['expected_gain']:.6f}, "
            f"P(improve)={best['prob_of_improvement']:.6f})"
        )

        observed_value = self.observe(best['observe_var'])
        if observed_value is None:
            return report

        evidence_after_observe = dict(self.known_evidence)
        best_two_switches, p_after_actions = self._find_best_two_switches(evidence_after_observe)
        if best_two_switches is None:
            print('No improving two-switch action found after observation.')
            return report

        switch1, switch2 = best_two_switches
        print(f'Planned action 1: switch {switch1} (invert state)')
        self._apply_switch(switch1)
        print(f'Planned action 2: switch {switch2} (invert state)')
        self._apply_switch(switch2)
        print(f'Estimated P(Crew_Status=safe) after plan: {p_after_actions:.6f}')

        return {
            **report,
            'observed': (best['observe_var'], observed_value),
            'actions': [(switch1, 'invert'), (switch2, 'invert')],
            'estimated_p_safe_after_actions': p_after_actions,
        }

    def run_strategy_switch_observe_switch(self):
        """Autonomous policy: one intervention, one observation, one intervention."""
        plan_report = self.choose_plan_switch_observe_switch()
        best_plan = plan_report['best_plan']
        if not best_plan:
            print('No profitable switch->observe->switch plan found.')
            return plan_report

        first_switch = best_plan['first_switch']
        print('=== Running Strategy: switch -> observe -> switch ===')
        print(
            f'Planned action 1: switch {first_switch} (invert state) '
            f"(expected final P(Crew_Status=safe)={best_plan['expected_prob']:.6f})"
        )
        self._apply_switch(first_switch)

        observed = None
        if best_plan['observe_var'] is not None:
            print(f"Planned observe: {best_plan['observe_var']}")
            observed_value = self.observe(best_plan['observe_var'])
            observed = (best_plan['observe_var'], observed_value)

        second_action, second_prob = self._find_best_single_switch(
            dict(self.known_evidence),
            excluded_switches={first_switch},
        )
        if second_action is None:
            print('No second switch improves the current state estimate.')
            return {
                **plan_report,
                'observed': observed,
                'actions': [(first_switch, 'invert')],
                'estimated_p_safe_after_actions': self._p_safe(dict(self.known_evidence)),
            }

        second_switch = second_action
        print(f'Planned action 2: switch {second_switch} (invert state)')
        self._apply_switch(second_switch)
        print(f'Estimated P(Crew_Status=safe) after plan: {second_prob:.6f}')

        return {
            **plan_report,
            'observed': observed,
            'actions': [(first_switch, 'invert'), (second_switch, 'invert')],
            'estimated_p_safe_after_actions': second_prob,
        }

    def run_strategy_observe_observe_observe_switch(self):
        """Autonomous policy: three observations followed by one intervention."""
        print('=== Running Strategy: observe -> observe -> observe -> switch ===')

        observed_steps = []
        for step in [1, 2, 3]:
            available_observes = tuple(sorted(self.obs_vars - set(self.known_evidence.keys())))
            if not available_observes:
                print('No observable variables left before completing 3 observes.')
                break

            best_observe = self._choose_best_observe_for_remaining_depth(
                dict(self.known_evidence),
                available_observes,
                remaining_observes=4 - step,
                excluded_switches=set(),
            )
            if not best_observe:
                print('No suitable observe action found.')
                break

            print(
                f"Planned observe {step}: {best_observe['observe_var']} "
                f"(expected gain={best_observe['expected_gain']:.6f}, "
                f"P(improve)={best_observe['prob_of_improvement']:.6f})"
            )
            value = self.observe(best_observe['observe_var'])
            observed_steps.append((best_observe['observe_var'], value))

        best_switch_action, best_prob = self._find_best_single_switch(dict(self.known_evidence))
        if best_switch_action is None:
            print('No switch action improves the current estimate after observations.')
            return {
                'observed': observed_steps,
                'actions': [],
                'estimated_p_safe_after_actions': self._p_safe(dict(self.known_evidence)),
            }

        switch = best_switch_action
        print(f'Planned final action: switch {switch} (invert state)')
        self._apply_switch(switch)
        print(f'Estimated P(Crew_Status=safe) after plan: {best_prob:.6f}')

        return {
            'observed': observed_steps,
            'actions': [(switch, 'invert')],
            'estimated_p_safe_after_actions': best_prob,
        }

    def run_strategy(self, strategy_name):
        """Unified entrypoint for autonomous strategies."""
        if strategy_name == 'observe_switch_switch':
            return self.run_strategy_observe_switch_switch()
        if strategy_name == 'switch_observe_switch':
            return self.run_strategy_switch_observe_switch()
        if strategy_name == 'observe_observe_observe_switch':
            return self.run_strategy_observe_observe_observe_switch()
        raise ValueError(
            "Unknown strategy_name. Use one of: "
            "observe_switch_switch, switch_observe_switch, observe_observe_observe_switch"
        )

    def estimate_strategy_priors(self):
        """Estimate prior success probability for each available strategy."""
        evidence = dict(self.known_evidence)
        priors = {}

        # observe -> switch -> switch
        oss_report = self.choose_best_observation_for_oss()
        if oss_report['best_observe'] is None:
            priors['observe_switch_switch'] = self._p_safe(evidence)
        else:
            priors['observe_switch_switch'] = float(
                oss_report['best_observe']['expected_prob_after_observe']
            )

        # switch -> observe -> switch
        sos_report = self.choose_plan_switch_observe_switch()
        if sos_report['best_plan'] is None:
            priors['switch_observe_switch'] = float(sos_report['baseline_prob'])
        else:
            priors['switch_observe_switch'] = float(sos_report['best_plan']['expected_prob'])

        # observe -> observe -> observe -> switch
        available_observes = tuple(sorted(self.obs_vars - set(evidence.keys())))
        ooos_depth = min(3, len(available_observes))
        priors['observe_observe_observe_switch'] = float(
            self._best_expected_prob_after_k_observes_then_switch(
                evidence,
                available_observes,
                remaining_observes=ooos_depth,
                excluded_switches=set(),
                cache={},
            )
        )

        return priors

    def choose_best_strategy_by_prior(self):
        """Choose strategy with maximum prior expected success probability."""
        priors = self.estimate_strategy_priors()
        best_strategy = max(priors, key=priors.get)
        return {
            'priors': priors,
            'best_strategy': best_strategy,
            'best_prior': priors[best_strategy],
        }

    def run_best_strategy_by_prior(self):
        """Estimate priors, print comparison, and execute the best strategy."""
        report = self.choose_best_strategy_by_prior()

        print('=== Strategy Prior Comparison ===')
        for strategy_name in [
            'observe_switch_switch',
            'switch_observe_switch',
            'observe_observe_observe_switch',
        ]:
            print(
                f"{strategy_name}: "
                f"P(Crew_Status=safe)~{report['priors'][strategy_name]:.6f}"
            )
        print(
            f"Selected strategy: {report['best_strategy']} "
            f"(prior={report['best_prior']:.6f})"
        )

        result = self.run_strategy(report['best_strategy'])
        return {
            **report,
            'execution_result': result,
        }


if __name__ == '__main__':
    bn = BIFReader('spaceship.bif').get_model()

    # ini_state = {'O2_Level': 'low', 'Temperature': 'cold', 'Alert_System': 'warning', 'Diagnosis': 'nominal'}
    ini_state = {'O2_Level': 'low', 'Alert_System': 'warning', 'Diagnosis': 'anomaly'}
    # ini_state = {'Temperature': 'hot', 'Alert_System': 'silent', 'Diagnosis': 'nominal'}
    # ini_state = {'AI_Test': 'pass', 'Temperature': 'hot', 'O2_Level': 'low', 'Alert_System': 'silent', 'Diagnosis': 'nominal'}
    # ini_state = {'CO2_Level': 'high', 'Alert_System': 'silent', 'Porthole': 'danger'}
    run_seed = random.SystemRandom().randrange(0, 2**32)
    print(f'Run seed: {run_seed}')
    env = SpaceOdysseySimulator(bn, initial_state=ini_state, seed=run_seed)

    env.run_best_strategy_by_prior()
    # env.run_strategy('observe_switch_switch')
    env.finish()
