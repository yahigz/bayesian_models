from __future__ import annotations

import argparse
import string
import sys
from dataclasses import dataclass
from typing import Sequence


DEFAULT_ROWS = 6
DEFAULT_COLS = 6
DEFAULT_FLEET = ((3, 1), (2, 2), (1, 3))


@dataclass(frozen=True, slots=True)
class ShipSpec:
	size: int
	count: int


@dataclass(frozen=True, slots=True)
class Placement:
	size: int
	cells: tuple[int, ...]
	occupied_mask: int
	forbidden_mask: int


@dataclass(frozen=True, slots=True)
class GameConfig:
	rows: int = DEFAULT_ROWS
	cols: int = DEFAULT_COLS
	fleet: tuple[ShipSpec, ...] = tuple(ShipSpec(size, count) for size, count in DEFAULT_FLEET)

	@property
	def cell_count(self) -> int:
		return self.rows * self.cols

	@property
	def total_ship_cells(self) -> int:
		return sum(spec.size * spec.count for spec in self.fleet)


@dataclass(slots=True)
class Evidence:
	shot_mask: int = 0
	hit_mask: int = 0
	miss_mask: int = 0


@dataclass(slots=True)
class GameBoard:
	ships: list[Placement]
	occupied_mask: int
	cell_to_ship: dict[int, int]


class BayesianBattleshipSolver:
	def __init__(self, config: GameConfig) -> None:
		self.config = config
		self.cell_names = [self._index_to_name(index) for index in range(config.cell_count)]
		self.index_by_name = {name: index for index, name in enumerate(self.cell_names)}
		self.neighbor_masks = [self._neighbor_mask(index) for index in range(config.cell_count)]
		self.placements_by_size = {
			spec.size: self._build_placements(spec.size) for spec in self.config.fleet
		}
		self.group_specs = sorted(self.config.fleet, key=lambda spec: (-spec.size, spec.count))

	def _row_label(self, row_index: int) -> str:
		value = row_index + 1
		label = ""
		while value:
			value, remainder = divmod(value - 1, 26)
			label = chr(ord("A") + remainder) + label
		return label

	def _row_index_from_label(self, label: str) -> int:
		value = 0
		for char in label:
			value = value * 26 + (ord(char) - ord("A") + 1)
		return value - 1

	def _index_to_name(self, index: int) -> str:
		row, col = divmod(index, self.config.cols)
		return f"{self._row_label(row)}{col + 1}"

	def _name_to_index(self, name: str) -> int:
		cleaned = name.strip().upper()
		if not cleaned:
			raise ValueError("empty cell name")

		row_part: list[str] = []
		col_part: list[str] = []
		for char in cleaned:
			if char.isalpha():
				if col_part:
					raise ValueError(f"invalid cell name '{name}'")
				row_part.append(char)
			elif char.isdigit():
				col_part.append(char)
			else:
				raise ValueError(f"invalid cell name '{name}'")

		if not row_part or not col_part:
			raise ValueError(f"invalid cell name '{name}'")

		row = self._row_index_from_label("".join(row_part))
		col = int("".join(col_part)) - 1
		if not (0 <= row < self.config.rows and 0 <= col < self.config.cols):
			raise ValueError(f"cell '{name}' is outside the {self.config.rows}x{self.config.cols} board")
		return row * self.config.cols + col

	def _bit(self, index: int) -> int:
		return 1 << index

	def _neighbor_mask(self, index: int) -> int:
		row, col = divmod(index, self.config.cols)
		mask = 0
		for d_row in (-1, 0, 1):
			for d_col in (-1, 0, 1):
				new_row = row + d_row
				new_col = col + d_col
				if 0 <= new_row < self.config.rows and 0 <= new_col < self.config.cols:
					mask |= self._bit(new_row * self.config.cols + new_col)
		return mask

	def _build_placements(self, size: int) -> list[Placement]:
		placements: list[Placement] = []
		seen: set[int] = set()

		def add_placement(cell_indices: list[int]) -> None:
			occupied_mask = 0
			forbidden_mask = 0
			for index in cell_indices:
				occupied_mask |= self._bit(index)
				forbidden_mask |= self.neighbor_masks[index]
			if occupied_mask not in seen:
				seen.add(occupied_mask)
				placements.append(
					Placement(
						size=size,
						cells=tuple(cell_indices),
						occupied_mask=occupied_mask,
						forbidden_mask=forbidden_mask,
					)
				)

		for row in range(self.config.rows):
			for col in range(self.config.cols):
				if col + size <= self.config.cols:
					cells = [row * self.config.cols + (col + offset) for offset in range(size)]
					add_placement(cells)
				if row + size <= self.config.rows:
					cells = [(row + offset) * self.config.cols + col for offset in range(size)]
					add_placement(cells)

		placements.sort(key=lambda placement: placement.occupied_mask)
		return placements

	def parse_ship_line(self, line: str) -> Placement:
		tokens = [token for token in line.replace(",", " ").split() if token]
		if not tokens:
			raise ValueError("ship line is empty")

		indices = [self._name_to_index(token) for token in tokens]
		if len(set(indices)) != len(indices):
			raise ValueError(f"duplicate cells in ship definition: '{line}'")

		if len(indices) == 1:
			index = indices[0]
			return Placement(
				size=1,
				cells=(index,),
				occupied_mask=self._bit(index),
				forbidden_mask=self.neighbor_masks[index],
			)

		ordered = sorted(indices)
		row_set = {divmod(index, self.config.cols)[0] for index in ordered}
		col_set = {divmod(index, self.config.cols)[1] for index in ordered}
		if len(row_set) != 1 and len(col_set) != 1:
			raise ValueError(f"ship must be straight: '{line}'")

		if len(row_set) == 1:
			row = next(iter(row_set))
			start_col = divmod(ordered[0], self.config.cols)[1]
			expected = [row * self.config.cols + col for col in range(start_col, start_col + len(ordered))]
			if ordered != expected:
				raise ValueError(f"cells must be consecutive: '{line}'")
		else:
			col = next(iter(col_set))
			start_row = divmod(ordered[0], self.config.cols)[0]
			expected = [(start_row + offset) * self.config.cols + col for offset in range(len(ordered))]
			if ordered != expected:
				raise ValueError(f"cells must be consecutive: '{line}'")

		occupied_mask = 0
		forbidden_mask = 0
		for index in ordered:
			occupied_mask |= self._bit(index)
			forbidden_mask |= self.neighbor_masks[index]

		return Placement(
			size=len(ordered),
			cells=tuple(ordered),
			occupied_mask=occupied_mask,
			forbidden_mask=forbidden_mask,
		)

	def parse_ship_layout(self, text: str) -> list[Placement]:
		lines = [line.strip() for line in text.replace(";", "\n").splitlines() if line.strip()]
		if not lines:
			raise ValueError("no ships were provided")
		return [self.parse_ship_line(line) for line in lines]

	def validate_board(self, ships: Sequence[Placement]) -> GameBoard:
		expected = {spec.size: spec.count for spec in self.config.fleet}
		actual: dict[int, int] = {}
		occupied_mask = 0
		forbidden_mask = 0
		cell_to_ship: dict[int, int] = {}

		for ship_index, ship in enumerate(ships):
			actual[ship.size] = actual.get(ship.size, 0) + 1
			if ship.size not in expected:
				raise ValueError(f"unexpected ship size {ship.size}")
			if actual[ship.size] > expected[ship.size]:
				raise ValueError(f"too many ships of size {ship.size}")
			if ship.occupied_mask & occupied_mask:
				raise ValueError("ships overlap")
			if ship.occupied_mask & forbidden_mask:
				raise ValueError("ships touch each other")
			occupied_mask |= ship.occupied_mask
			forbidden_mask |= ship.forbidden_mask
			for cell_index in ship.cells:
				cell_to_ship[cell_index] = ship_index

		for size, count in expected.items():
			if actual.get(size, 0) != count:
				raise ValueError(f"expected {count} ship(s) of size {size}, got {actual.get(size, 0)}")

		if len(cell_to_ship) != self.config.total_ship_cells:
			raise ValueError("ship cells do not match the expected fleet size")

		return GameBoard(ships=list(ships), occupied_mask=occupied_mask, cell_to_ship=cell_to_ship)

	def compute_posteriors(self, evidence: Evidence) -> tuple[list[float], int]:
		counts = [0] * self.config.cell_count
		total_states = 0

		filtered_by_size: dict[int, list[Placement]] = {}
		for size, placements in self.placements_by_size.items():
			filtered_by_size[size] = [placement for placement in placements if not (placement.occupied_mask & evidence.miss_mask)]

		remaining_cells_by_group: list[int] = []
		running = 0
		for spec in reversed(self.group_specs):
			running += spec.size * spec.count
			remaining_cells_by_group.append(running)
		remaining_cells_by_group.reverse()

		def collect_counts(mask: int) -> None:
			nonlocal total_states
			total_states += 1
			while mask:
				bit = mask & -mask
				counts[bit.bit_length() - 1] += 1
				mask ^= bit

		def dfs_group(group_index: int, occupied_mask: int, forbidden_mask: int) -> None:
			if group_index == len(self.group_specs):
				if evidence.hit_mask & ~occupied_mask:
					return
				collect_counts(occupied_mask)
				return

			spec = self.group_specs[group_index]
			remaining_hits = (evidence.hit_mask & ~occupied_mask).bit_count()
			if remaining_hits > remaining_cells_by_group[group_index]:
				return

			candidates = filtered_by_size[spec.size]

			def choose_ship(ship_index: int, start_index: int, current_occupied: int, current_forbidden: int) -> None:
				if ship_index == spec.count:
					dfs_group(group_index + 1, current_occupied, current_forbidden)
					return

				for candidate_index in range(start_index, len(candidates)):
					placement = candidates[candidate_index]
					if placement.occupied_mask & (current_occupied | current_forbidden):
						continue
					choose_ship(
						ship_index + 1,
						candidate_index + 1,
						current_occupied | placement.occupied_mask,
						current_forbidden | placement.forbidden_mask,
					)

			choose_ship(0, 0, occupied_mask, forbidden_mask)

		dfs_group(0, 0, 0)

		if total_states == 0:
			return [0.0] * self.config.cell_count, 0

		probabilities = [count / total_states for count in counts]
		return probabilities, total_states

	def choose_move(self, probabilities: Sequence[float], evidence: Evidence) -> tuple[int, float]:
		best_index = -1
		best_probability = -1.0
		for index, probability in enumerate(probabilities):
			if evidence.shot_mask & self._bit(index):
				continue
			if probability > best_probability or (probability == best_probability and index < best_index):
				best_index = index
				best_probability = probability
		if best_index < 0:
			raise RuntimeError("no legal moves remain")
		return best_index, best_probability

	def render_heatmap(self, probabilities: Sequence[float], evidence: Evidence) -> str:
		header = ["   "]
		for col in range(self.config.cols):
			header.append(f"{col + 1:>6}")
		lines = ["".join(header)]

		for row in range(self.config.rows):
			parts = [f"{self._row_label(row):>2} "]
			for col in range(self.config.cols):
				index = row * self.config.cols + col
				bit = self._bit(index)
				if evidence.hit_mask & bit:
					text = " HIT "
				elif evidence.miss_mask & bit:
					text = " MISS"
				else:
					text = self._color_probability(probabilities[index])
				parts.append(text)
			lines.append("".join(parts))
		return "\n".join(lines)

	def _color_probability(self, probability: float) -> str:
		label = f"{probability * 100:4.0f}%"
		if not sys.stdout.isatty():
			return f" {label}"
		red, green, blue = self._gradient_rgb(probability)
		text_red, text_green, text_blue = self._contrast_rgb(red, green, blue)
		return f"\033[48;2;{red};{green};{blue}m\033[38;2;{text_red};{text_green};{text_blue}m{label}\033[0m"

	def _gradient_rgb(self, probability: float) -> tuple[int, int, int]:
		stops = (
			(0.0, (22, 58, 148)),
			(0.5, (31, 164, 137)),
			(1.0, (232, 84, 47)),
		)
		clamped = max(0.0, min(1.0, probability))
		for left_index in range(len(stops) - 1):
			left_position, left_color = stops[left_index]
			right_position, right_color = stops[left_index + 1]
			if clamped <= right_position:
				span = right_position - left_position or 1.0
				ratio = (clamped - left_position) / span
				return tuple(
					int(round(left_channel + (right_channel - left_channel) * ratio))
					for left_channel, right_channel in zip(left_color, right_color, strict=True)
				)
		return stops[-1][1]

	def _contrast_rgb(self, red: int, green: int, blue: int) -> tuple[int, int, int]:
		luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
		if luminance >= 140:
			return 20, 20, 20
		return 245, 245, 245


def parse_fleet(items: Sequence[str]) -> tuple[ShipSpec, ...]:
	if not items:
		return tuple(ShipSpec(size, count) for size, count in DEFAULT_FLEET)
	fleet: list[ShipSpec] = []
	for item in items:
		if ":" not in item:
			raise ValueError(f"invalid fleet item '{item}', expected size:count")
		size_text, count_text = item.split(":", 1)
		size = int(size_text)
		count = int(count_text)
		if size <= 0 or count <= 0:
			raise ValueError(f"fleet item must be positive: '{item}'")
		fleet.append(ShipSpec(size=size, count=count))
	return tuple(fleet)


def parse_layout_from_args(args: argparse.Namespace, solver: BayesianBattleshipSolver) -> list[Placement] | None:
	if args.ships:
		return solver.parse_ship_layout(args.ships)
	if args.ship_file:
		with open(args.ship_file, "r", encoding="utf-8") as file_handle:
			return solver.parse_ship_layout(file_handle.read())
	return None


def prompt_for_layout(solver: BayesianBattleshipSolver) -> list[Placement]:
	print("Enter ship placements, one ship per line.")
	print("Format: A1 A2 A3 or B2 C2 for a horizontal/vertical ship.")
	print("Separate ships with new lines or semicolons. Empty line ends input.")
	print("Expected fleet:", ", ".join(f"{spec.count}x{spec.size}" for spec in solver.config.fleet))

	lines: list[str] = []
	while True:
		line = input("> ").strip()
		if not line:
			break
		lines.append(line)
	return solver.parse_ship_layout("\n".join(lines))


def format_cell_name(solver: BayesianBattleshipSolver, index: int) -> str:
	return solver.cell_names[index]


def run_simulation(solver: BayesianBattleshipSolver, board: GameBoard) -> None:
	evidence = Evidence()
	turn = 1

	while evidence.hit_mask != board.occupied_mask:
		probabilities, total_states = solver.compute_posteriors(evidence)
		move_index, move_probability = solver.choose_move(probabilities, evidence)

		print()
		print(f"Turn {turn} | consistent board states: {total_states}")
		print(solver.render_heatmap(probabilities, evidence))
		print(f"Chosen move: {format_cell_name(solver, move_index)} with probability {move_probability:.2%}")

		bit = solver._bit(move_index)
		evidence.shot_mask |= bit
		if board.occupied_mask & bit:
			evidence.hit_mask |= bit
			ship_index = board.cell_to_ship[move_index]
			ship = board.ships[ship_index]
			if ship.occupied_mask & ~evidence.hit_mask == 0:
				print(f"Result: HIT, ship of size {ship.size} sunk")
			else:
				print("Result: HIT")
		else:
			evidence.miss_mask |= bit
			print("Result: MISS")

		turn += 1

	probabilities, total_states = solver.compute_posteriors(evidence)
	print()
	print(f"Finished in {turn - 1} turns")
	print(solver.render_heatmap(probabilities, evidence))
	print(f"All ships destroyed. Final consistent states: {total_states}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Bayesian battleship simulator")
	parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="board rows")
	parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help="board columns")
	parser.add_argument(
		"--ship",
		dest="ships_spec",
		action="append",
		default=[],
		help="fleet item in size:count format, e.g. 3:1",
	)
	parser.add_argument(
		"--ships",
		type=str,
		default=None,
		help="ship layout as one string, ships separated by semicolons",
	)
	parser.add_argument(
		"--ship-file",
		type=str,
		default=None,
		help="file with ship layout, one ship per line",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	config = GameConfig(rows=args.rows, cols=args.cols, fleet=parse_fleet(args.ships_spec))
	solver = BayesianBattleshipSolver(config)

	try:
		ships = parse_layout_from_args(args, solver)
		if ships is None:
			ships = prompt_for_layout(solver)
		board = solver.validate_board(ships)
	except Exception as exc:
		print(f"Invalid board: {exc}", file=sys.stderr)
		raise SystemExit(1) from exc

	run_simulation(solver, board)


if __name__ == "__main__":
	main()
