import math
from dataclasses import dataclass


ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ALPHABET_SET = set(ALPHABET)

FIRSTNAMES = ["david", "anton", "fred", "jim", "barry"]
SURNAMES = [
	"barber",
	"ilsung",
	"fox",
	"chain",
	"fitzwilliam",
	"quinceadams",
	"grafvonunterhosen",
]


@dataclass(frozen=True)
class State:
	kind: str  # "A", "B", "F", "S"
	idx: int = -1
	pos: int = -1


def log_emission(observed: str, intended: str) -> float:
	if observed == intended:
		return math.log(0.3)
	return math.log(0.7 / 25.0)


def build_states() -> tuple[list[State], dict[State, int]]:
	states: list[State] = [State("A"), State("B")]

	for i, name in enumerate(FIRSTNAMES):
		for pos in range(len(name)):
			states.append(State("F", i, pos))

	for j, surname in enumerate(SURNAMES):
		for pos in range(len(surname)):
			states.append(State("S", j, pos))

	state_to_idx = {s: i for i, s in enumerate(states)}
	return states, state_to_idx


def build_transitions(states: list[State], state_to_idx: dict[State, int]) -> list[list[tuple[int, float]]]:
	transitions: list[list[tuple[int, float]]] = [[] for _ in states]

	idx_a = state_to_idx[State("A")]
	idx_b = state_to_idx[State("B")]
	
	transitions[idx_a].append((idx_a, math.log(0.8)))
	for i, _name in enumerate(FIRSTNAMES):
		dst = state_to_idx[State("F", i, 0)]
		transitions[idx_a].append((dst, math.log(0.2 / len(FIRSTNAMES))))

	transitions[idx_b].append((idx_b, math.log(0.8)))
	for j, _surname in enumerate(SURNAMES):
		dst = state_to_idx[State("S", j, 0)]
		transitions[idx_b].append((dst, math.log(0.2 / len(SURNAMES))))

	for i, name in enumerate(FIRSTNAMES):
		for pos in range(len(name)):
			src = state_to_idx[State("F", i, pos)]
			if pos + 1 < len(name):
				dst = state_to_idx[State("F", i, pos + 1)]
				transitions[src].append((dst, 0.0))
			else:
				transitions[src].append((idx_b, 0.0))

	for j, surname in enumerate(SURNAMES):
		for pos in range(len(surname)):
			src = state_to_idx[State("S", j, pos)]
			if pos + 1 < len(surname):
				dst = state_to_idx[State("S", j, pos + 1)]
				transitions[src].append((dst, 0.0))
			else:
				transitions[src].append((idx_a, 0.0))

	return transitions


def emission_log_for_state(state: State, observed_char: str) -> float:
	if state.kind in ("A", "B"):
		return -math.log(26.0)

	if state.kind == "F":
		intended = FIRSTNAMES[state.idx][state.pos]
		return log_emission(observed_char, intended)

	intended = SURNAMES[state.idx][state.pos]
	return log_emission(observed_char, intended)


def decode_viterbi(observed: str) -> tuple[list[int], str]:
	states, state_to_idx = build_states()
	transitions = build_transitions(states, state_to_idx)
	n_states = len(states)
	t_max = len(observed)

	idx_a = state_to_idx[State("A")]
	neg_inf = float("-inf")

	dp_prev = [neg_inf] * n_states
	dp_prev[idx_a] = emission_log_for_state(states[idx_a], observed[0])

	backptr = [[-1] * n_states for _ in range(t_max)]

	incoming: list[list[tuple[int, float]]] = [[] for _ in range(n_states)]
	for src, edges in enumerate(transitions):
		for dst, log_p in edges:
			incoming[dst].append((src, log_p))

	for t in range(1, t_max):
		ch = observed[t]
		dp_cur = [neg_inf] * n_states
		for dst in range(n_states):
			best_val = neg_inf
			best_src = -1
			for src, log_tr in incoming[dst]:
				prev = dp_prev[src]
				if prev == neg_inf:
					continue
				val = prev + log_tr
				if val > best_val:
					best_val = val
					best_src = src

			if best_src != -1:
				dp_cur[dst] = best_val + emission_log_for_state(states[dst], ch)
				backptr[t][dst] = best_src

		dp_prev = dp_cur

	last_state = max(range(n_states), key=lambda s: dp_prev[s])
	path = [0] * t_max
	path[-1] = last_state
	for t in range(t_max - 1, 0, -1):
		path[t - 1] = backptr[t][path[t]]

	clean_chars: list[str] = []
	for t, s_idx in enumerate(path):
		st = states[s_idx]
		if st.kind in ("A", "B"):
			clean_chars.append(observed[t])
		elif st.kind == "F":
			clean_chars.append(FIRSTNAMES[st.idx][st.pos])
		else:
			clean_chars.append(SURNAMES[st.idx][st.pos])

	return path, "".join(clean_chars)


def count_pairs(path: list[int]) -> list[list[int]]:
	states, _ = build_states()
	m = [[0 for _ in SURNAMES] for _ in FIRSTNAMES]

	current_firstname_idx = None
	prev_kind = None

	for s_idx in path:
		st = states[s_idx]
		if st.kind == "F" and st.pos == 0 and prev_kind != "F":
			current_firstname_idx = st.idx

		if st.kind == "S" and st.pos == 0 and prev_kind != "S":
			if current_firstname_idx is not None:
				m[current_firstname_idx][st.idx] += 1
			current_firstname_idx = None

		prev_kind = st.kind

	return m


def read_noisy_string(path: str) -> str:
	with open(path, "r", encoding="utf-8") as f:
		raw = f.read().strip().lower()

	cleaned = "".join(ch for ch in raw if ch in ALPHABET_SET)
	if not cleaned:
		raise ValueError("No valid lowercase letters found in noisy string file")
	return cleaned


def print_matrix(m: list[list[int]]) -> None:
	print("Firstnames:", ", ".join(FIRSTNAMES))
	print("Surnames:", ", ".join(SURNAMES))
	print("\nm(i, j) counts (rows=firstnames, cols=surnames):")

	header = [" " * 14] + [f"{s:>16}" for s in SURNAMES]
	print("".join(header))
	for i, row in enumerate(m):
		row_str = [f"{FIRSTNAMES[i]:>14}"] + [f"{v:>16d}" for v in row]
		print("".join(row_str))


def main() -> None:
	observed = read_noisy_string("task5_8/noisystring.txt")
	path, clean = decode_viterbi(observed)
	m = count_pairs(path)

	print(f"Observed length: {len(observed)}")
	print(f"Decoded clean length: {len(clean)}")
	print_matrix(m)


if __name__ == "__main__":
	main()
