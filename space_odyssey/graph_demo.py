"""
Демонстрация графа MDP для симуляции космического корабля.

Граф представляет:
- Состояние: множество известных (obs_vars) 
- Действие: интервенция на переменную из do_vars
- Переход: действие делает неизвестными все obs_vars потомки изменяемой переменной
"""

from pgmpy.readwrite import BIFReader
from env import SpaceOdysseySimulator, OBS_VARS, DO_VARS

# Загрузить модель
reader = BIFReader("spaceship.bif")
model = reader.get_model()

# Создать симулятор
sim = SpaceOdysseySimulator(model, seed=42)

print("\n" + "="*80)
print("MARKOV DECISION PROCESS FOR SPACE ODYSSEY")
print("="*80)

# 1. Вывести информацию о структуре графа
sim.print_mdp_graph_info()

# 2. Вывести таблицу всех действий и их эффектов
sim.visualize_mdp_action_effects()

# 3. Пример переходов состояний
print("\n" + "="*80)
print("STATE TRANSITION EXAMPLES")
print("="*80)

initial_evidence = {
    'O2_Level': 'normal',
    'Temperature': 'normal',
    'Alert_System': 'silent'
}

sim.known_evidence = initial_evidence
print(f"\nInitial state: {sim.graph.get_state_description(initial_evidence)}")

# Показать, что происходит при действиях
for do_var in ['HAL_Switch', 'Thermostat', 'Manoeuvre']:
    next_state = sim.simulate_action_effect(initial_evidence, do_var, None)
    obs_desc = sorted(sim.get_obs_descendants_of_do_var(do_var))
    
    print(f"\n--- Action: {do_var} ---")
    print(f"Observable descendants that become unknown: {obs_desc if obs_desc else 'None'}")
    print(f"State after action: {sim.graph.get_state_description(next_state) or 'Empty'}")

# 4. Получить граф как NetworkX
print("\n" + "="*80)
print("GRAPH STATISTICS")
print("="*80)

G = sim.export_graph_as_nx()
print(f"Total nodes in DAG: {G.number_of_nodes()}")
print(f"Total edges in DAG: {G.number_of_edges()}")
print(f"Observable variables: {len(OBS_VARS)}")
print(f"Action variables: {len(DO_VARS)}")

# 5. Анализ потомков каждой переменной action
print("\n" + "="*80)
print("DESCENDANTS ANALYSIS")
print("="*80)

for do_var in sorted(DO_VARS):
    all_desc = sim.get_descendants_of_do_var(do_var)
    obs_desc = sim.get_obs_descendants_of_do_var(do_var)
    print(f"\n{do_var}:")
    print(f"  Total descendants: {len(all_desc)}")
    print(f"  Observable descendants: {len(obs_desc)}")
    if obs_desc:
        print(f"  Which are: {sorted(obs_desc)}")
