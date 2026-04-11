from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader
import logging
import random
import networkx as nx
from collections import deque
from typing import FrozenSet, Tuple, Dict, Set


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


class GraphMarkovDecisionProcess:
    """
    Граф состояний и действий для MDP.
    
    Состояние: frozenset известных (наблюдаемых) переменных и их значений
    Действие: (переменная из do_vars, значение)
    Переход: разновидность do-исчисления - действие делает неизвестными 
            все obs_vars, которые являются потомками изменяемой переменной
    """
    
    def __init__(self, model: DiscreteBayesianNetwork):
        """
        :param model: объект DiscreteBayesianNetwork с полной структурой ребер
        """
        self.model = model
        self.obs_vars = OBS_VARS
        self.do_vars = DO_VARS
        
        self.graph_nx = nx.DiGraph()
        for parent, child in model.edges():
            self.graph_nx.add_edge(parent, child)
        
        self._descendants_cache = {}
        for var in self.do_vars:
            self._descendants_cache[var] = self._compute_descendants(var)
    
    def _compute_descendants(self, node: str) -> Set[str]:
        """
        Найти всех потомков узла в DAG
        """
        descendants = set()
        queue = deque([node])
        visited = {node}
        
        while queue:
            current = queue.popleft()
            for successor in self.graph_nx.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    descendants.add(successor)
                    queue.append(successor)
        
        return descendants
    
    def get_obs_descendants(self, do_var: str) -> Set[str]:
        """
        Получить все obs_vars, которые являются потомками do_var
        """
        descendants = self._descendants_cache[do_var]
        return descendants & self.obs_vars
    
    def get_possible_actions(self) -> list:
        """
        Получить все возможные действия (переменная, значение).
        Значения берутся из реальных state_names узла, а не из индексов.
        """
        actions = []
        for var in self.do_vars:
            states = self.model.get_cpds(var).state_names[var]
            for value in states:
                actions.append((var, value))
        return actions
    
    def get_next_state(self, current_state: Dict, action: Tuple[str, str]) -> Dict:
        """
        Вычислить новое состояние после действия.
        Действие делает неизвестными все obs_vars-потомки do_var.
        
        :param current_state: dict {переменная: значение} для известных переменных
        :param action: (do_var, value)
        :return: новое состояние (новые неизвестные переменные удалены)
        """
        do_var, value = action
        new_state = dict(current_state)
        
        affected_obs = self.get_obs_descendants(do_var)
        
        for obs_var in affected_obs:
            new_state.pop(obs_var, None)
        
        return new_state
    
    def get_state_description(self, state: Dict) -> str:
        """
        Описать состояние в читаемом виде
        """
        if not state:
            return "No observations"
        items = [f"{var}={val}" for var, val in sorted(state.items())]
        return ", ".join(items)


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
        
        self.graph = GraphMarkovDecisionProcess(self.model)

        self.steps = 0
        self.action_points = self.start_action_points
        self.known_evidence = dict(initial_state) if initial_state else {}
        self.initial_state = initial_state
        self._state = self.model.simulate(
            n_samples=1,
            evidence=self.initial_state,
            seed=self._next_seed(),
            show_progress=False,
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
            show_progress=False,
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
            show_progress=False,
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

    def _state_prior_p_safe(self, evidence):
        """Prior probability P(Crew_Status=safe) for a state (known evidence)."""
        return self._p_safe(evidence)

    def _evidence_after_do(self, evidence, do_var, do_value):
        """Apply abstract graph transition for do-action over known evidence."""
        next_evidence = dict(evidence)
        for obs_var in self.get_obs_descendants_of_do_var(do_var):
            next_evidence.pop(obs_var, None)
        next_evidence[do_var] = do_value
        return next_evidence

    def _expected_prior_after_action(self, evidence, action):
        """
        Expected prior P(Crew_Status=safe) in the next graph node after action.
        Observe action is stochastic and uses expectation over observation posterior.
        Do action is deterministic in this abstract MDP graph.
        """
        kind = action['kind']
        if kind == 'observe':
            observe_var = action['var']
            posterior = self.infer.query(variables=[observe_var], evidence=evidence)
            state_to_index = {
                state: idx for idx, state in enumerate(posterior.state_names[observe_var])
            }

            expected_prob = 0.0
            for state in posterior.state_names[observe_var]:
                state_prob = float(posterior.values[state_to_index[state]])
                next_evidence = {**evidence, observe_var: state}
                expected_prob += state_prob * self._state_prior_p_safe(next_evidence)
            return expected_prob

        if kind == 'do':
            next_evidence = self._evidence_after_do(
                evidence,
                action['var'],
                action['value'],
            )
            return self._state_prior_p_safe(next_evidence)

        raise ValueError(f"Unknown action kind: {kind}")

    def _build_available_actions(self, evidence, action_points):
        """Build all feasible actions under the current points budget."""
        actions = []

        if action_points >= 1:
            unknown_obs_vars = sorted(self.obs_vars - set(evidence.keys()))
            for obs_var in unknown_obs_vars:
                actions.append({
                    'kind': 'observe',
                    'var': obs_var,
                    'cost': 1,
                })

        if action_points >= 2:
            for do_var in sorted(self.do_vars):
                idx_to_state = self._get_state_mapping(do_var)[0]
                for do_value in idx_to_state.values():
                    actions.append({
                        'kind': 'do',
                        'var': do_var,
                        'value': do_value,
                        'cost': 2,
                    })

        return actions

    def choose_best_next_action_by_prior(self):
        """
        Choose the next graph edge from current node by maximum prior score.
        Node: known evidence; edge: observe/do action with cost constraints.
        """
        evidence = dict(self.known_evidence)
        current_prior = self._state_prior_p_safe(evidence)
        candidates = self._build_available_actions(evidence, self.action_points)

        if not candidates:
            return {
                'current_prior': current_prior,
                'best': None,
                'ranked': [],
            }

        ranked = []
        for action in candidates:
            expected_prior = self._expected_prior_after_action(evidence, action)
            ranked.append({
                'action': action,
                'expected_prior': expected_prior,
                'gain': expected_prior - current_prior,
            })

        # Maximize expected prior; then maximize gain; then prefer cheaper action.
        ranked.sort(
            key=lambda x: (
                x['expected_prior'],
                x['gain'],
                -x['action']['cost'],
            ),
            reverse=True,
        )

        return {
            'current_prior': current_prior,
            'best': ranked[0],
            'ranked': ranked,
        }

    def run_greedy_prior_graph_policy(self):
        """
        Execute policy: from each node move to action with maximal expected prior.
        Continues until no actions are feasible or points are exhausted.
        """
        print('=== Running Greedy Prior Graph Policy ===')
        trajectory = []

        while self.action_points > 0:
            decision = self.choose_best_next_action_by_prior()
            best = decision['best']
            if best is None:
                print('No feasible actions left.')
                break

            action = best['action']
            if action['kind'] == 'observe':
                print(
                    f"Choose observe {action['var']} "
                    f"(expected prior={best['expected_prior']:.6f}, "
                    f"gain={best['gain']:.6f})"
                )
                value = self.observe(action['var'])
                trajectory.append({
                    'action': action,
                    'observed_value': value,
                    'expected_prior': best['expected_prior'],
                    'gain': best['gain'],
                })
                if value is None:
                    break
                continue

            print(
                f"Choose do {action['var']}={action['value']} "
                f"(expected prior={best['expected_prior']:.6f}, "
                f"gain={best['gain']:.6f})"
            )
            self.act(action['var'], action['value'])
            trajectory.append({
                'action': action,
                'expected_prior': best['expected_prior'],
                'gain': best['gain'],
            })

        final_prior = self._state_prior_p_safe(dict(self.known_evidence))
        print(f'Final prior P(Crew_Status=safe): {final_prior:.6f}')
        return {
            'trajectory': trajectory,
            'final_prior': final_prior,
            'remaining_action_points': self.action_points,
        }

    def get_descendants_of_do_var(self, do_var: str) -> Set[str]:
        """
        Получить всех потомков переменной action в DAG.
        
        :param do_var: переменная из DO_VARS
        :return: множество всех переменных, зависимых от do_var
        """
        if do_var not in self.do_vars:
            raise ValueError(f"{do_var} not in DO_VARS")
        return self.graph._descendants_cache[do_var]
    
    def get_obs_descendants_of_do_var(self, do_var: str) -> Set[str]:
        """
        Получить только наблюдаемые потомки переменной action.
        Именно эти переменные становятся неизвестными после действия.
        
        :param do_var: переменная из DO_VARS
        :return: множество obs_vars, которые являются потомками do_var
        """
        if do_var not in self.do_vars:
            raise ValueError(f"{do_var} not in DO_VARS")
        return self.graph.get_obs_descendants(do_var)
    
    def print_mdp_graph_info(self):
        """
        Вывести информацию о структуре графа MDP.
        Для каждой переменной action показать, какие obs_vars будут неизвестны после её применения.
        """
        print("=" * 70)
        print("MARKOV DECISION PROCESS - GRAPH STRUCTURE")
        print("=" * 70)
        print(f"\nObservable variables (obs_vars): {sorted(self.obs_vars)}")
        print(f"Action variables (do_vars): {sorted(self.do_vars)}")
        print("\n" + "-" * 70)
        print("EFFECT OF EACH ACTION:")
        print("-" * 70)
        
        for do_var in sorted(self.do_vars):
            all_descendants = self.get_descendants_of_do_var(do_var)
            obs_descendants = self.get_obs_descendants_of_do_var(do_var)
            
            print(f"\n{do_var}:")
            print(f"  All descendants: {sorted(all_descendants) if all_descendants else 'None'}")
            print(f"  Observable descendants (become unknown after action):")
            if obs_descendants:
                for obs in sorted(obs_descendants):
                    print(f"    - {obs}")
            else:
                print(f"    None")
    
    def simulate_action_effect(self, state: Dict, do_var: str, value) -> Dict:
        """
        Симулировать эффект действия на текущее состояние.
        
        State: множество известных переменных
        Action: (do_var, value)
        Next State: state с удаленными obs_descendants(do_var)
        
        :param state: текущее состояние (dict с известными переменными)
        :param do_var: переменная для интервенции
        :param value: значение для установки
        :return: новое состояние после действия
        """
        if do_var not in self.do_vars:
            raise ValueError(f"{do_var} not in DO_VARS")
        
        return self.graph.get_next_state(state, (do_var, value))
    
    def print_state_transition_example(self, do_var: str):
        """
        Показать пример перехода состояния при применении действия.
        
        :param do_var: переменная для интервенции
        """
        current_state = dict(self.known_evidence)
        next_state = self.simulate_action_effect(current_state, do_var, None)
        affected = current_state.keys() - next_state.keys()
        
        print(f"\nAction: {do_var}")
        print(f"Current state: {self.graph.get_state_description(current_state) or 'Empty'}")
        print(f"After action {do_var}:")
        print(f"  Variables that become unknown: {sorted(affected) if affected else 'None'}")
        print(f"  New state: {self.graph.get_state_description(next_state) or 'Empty'}")
    
    def export_graph_as_nx(self) -> nx.DiGraph:
        """
        Экспортировать граф Байесовской сети как NetworkX DiGraph.
        Полезно для дальнейшего анализа структуры.
        """
        return self.graph.graph_nx.copy()
    
    def visualize_mdp_action_effects(self):
        """
        Вывести таблицу всех действий и их эффектов на obs_vars.
        """
        print("\n" + "=" * 90)
        print("MDP ACTION EFFECTS TABLE")
        print("=" * 90)
        
        # Найти максимальную длину для форматирования
        max_do_len = max(len(v) for v in self.do_vars)
        
        for do_var in sorted(self.do_vars):
            obs_desc = sorted(self.get_obs_descendants_of_do_var(do_var))
            print(f"\n{do_var:<{max_do_len}} -> invalidates: {', '.join(obs_desc) if obs_desc else '(none)'}")



if __name__ == '__main__':
    bn = BIFReader('spaceship.bif').get_model()

    # ini_state = {'O2_Level': 'low', 'Temperature': 'cold', 'Alert_System': 'warning', 'Diagnosis': 'nominal'}
    # ini_state = {'O2_Level': 'low', 'Alert_System': 'warning', 'Diagnosis': 'anomaly'}
    # ini_state = {'Temperature': 'hot', 'Alert_System': 'silent', 'Diagnosis': 'nominal'}
    ini_state = {'AI_Test': 'pass', 'Temperature': 'hot', 'O2_Level': 'low', 'Alert_System': 'silent', 'Diagnosis': 'nominal'}
    # ini_state = {'CO2_Level': 'high', 'Alert_System': 'silent', 'Porthole': 'danger'}
    run_seed = random.SystemRandom().randrange(0, 2**32)
    print(f'Run seed: {run_seed}')
    env = SpaceOdysseySimulator(bn, initial_state=ini_state, seed=run_seed)

    env.run_greedy_prior_graph_policy()
    env.finish()
