from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Rectangle

from main import BayesianBattleshipSolver, DEFAULT_COLS, DEFAULT_ROWS, Evidence, GameBoard, GameConfig, Placement, parse_fleet


SHIP_TOKENS = {"x", "X", "1", "*", "#"}
WATER_TOKENS = {"0", ".", "-"}


@dataclass(frozen=True, slots=True)
class ParsedMatrix:
    rows: list[list[int]]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return len(self.rows[0]) if self.rows else 0


@dataclass(frozen=True, slots=True)
class TurnSnapshot:
    turn: int
    probabilities: list[float]
    total_states: int
    move_index: int
    move_probability: float
    evidence: Evidence


class SimulationViewer:
    def __init__(self, solver: BayesianBattleshipSolver, board: GameBoard, snapshots: Sequence[TurnSnapshot]) -> None:
        self.solver = solver
        self.board = board
        self.snapshots = list(snapshots)
        self.index = 0

        sns.set_theme(style="white", context="notebook")
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    def show(self) -> None:
        plt.show()

    def _on_key(self, event) -> None:
        if event.key in {"right", "down", "pagedown", "space"}:
            self.index = min(self.index + 1, len(self.snapshots) - 1)
            self._draw()
        elif event.key in {"left", "up", "pageup"}:
            self.index = max(self.index - 1, 0)
            self._draw()
        elif event.key == "home":
            self.index = 0
            self._draw()
        elif event.key == "end":
            self.index = len(self.snapshots) - 1
            self._draw()

    def _draw(self) -> None:
        snapshot = self.snapshots[self.index]
        self.ax.clear()

        probability_matrix = np.array(snapshot.probabilities, dtype=float).reshape(
            self.solver.config.rows, self.solver.config.cols
        )
        ship_matrix = np.zeros_like(probability_matrix)
        hit_matrix = np.zeros_like(probability_matrix)
        miss_matrix = np.zeros_like(probability_matrix)
        for cell_index in self.board.cell_to_ship:
            row, col = divmod(cell_index, self.solver.config.cols)
            ship_matrix[row, col] = 1.0
        for index in range(self.solver.config.cell_count):
            row, col = divmod(index, self.solver.config.cols)
            bit = self.solver._bit(index)
            if snapshot.evidence.hit_mask & bit:
                hit_matrix[row, col] = 1.0
            elif snapshot.evidence.miss_mask & bit:
                miss_matrix[row, col] = 1.0

        probability_cmap = LinearSegmentedColormap.from_list(
            "probability_gradient",
            ["#f4f8ff", "#cfe7ff", "#8ec7ff", "#4ca3ff", "#38c6b0", "#f1d45a", "#f0a43d", "#d9791e"],
        )
        norm = PowerNorm(gamma=0.5, vmin=float(np.min(probability_matrix)), vmax=float(np.max(probability_matrix)))
        if abs(norm.vmax - norm.vmin) < 1e-9:
            norm = PowerNorm(gamma=0.5, vmin=float(norm.vmin), vmax=float(norm.vmin + 1.0))

        sns.heatmap(
            probability_matrix,
            ax=self.ax,
            cmap=probability_cmap,
            norm=norm,
            cbar=False,
            square=True,
            linewidths=1,
            linecolor="#e6e6e6",
            annot=False,
        )

        self._draw_state_overlay(hit_matrix, miss_matrix)

        self._draw_board_state(snapshot, ship_matrix)
        self._draw_move_marker(snapshot)

        self.ax.set_title(
            f"Turn {snapshot.turn} | states: {snapshot.total_states} | move: {self.solver.cell_names[snapshot.move_index]}"
        )
        self.ax.set_xlabel("Columns")
        self.ax.set_ylabel("Rows")
        self.ax.set_xticks(np.arange(self.solver.config.cols) + 0.5)
        self.ax.set_xticklabels([str(col + 1) for col in range(self.solver.config.cols)])
        self.ax.set_yticks(np.arange(self.solver.config.rows) + 0.5)
        self.ax.set_yticklabels([self.solver._row_label(row) for row in range(self.solver.config.rows)], rotation=0)
        self.ax.tick_params(axis="x", rotation=0)
        self.ax.set_aspect("equal")

        self.figure.tight_layout()
        self.figure.canvas.draw_idle()

    def _draw_board_state(self, snapshot: TurnSnapshot, ship_matrix: np.ndarray) -> None:
        for row in range(self.solver.config.rows):
            for col in range(self.solver.config.cols):
                index = row * self.solver.config.cols + col
                bit = self.solver._bit(index)

                if snapshot.evidence.miss_mask & bit:
                    continue

                if ship_matrix[row, col] > 0:
                    probability_label = f"{snapshot.probabilities[index] * 100:.0f}%"
                    probability_color = "white" if snapshot.evidence.hit_mask & bit else self._text_color_for_probability(snapshot.probabilities[index])
                    x_color = "white" if snapshot.evidence.hit_mask & bit else "#101010"

                    self.ax.text(
                        col + 0.5,
                        row + 0.28,
                        probability_label,
                        ha="center",
                        va="center",
                        color=probability_color,
                        fontsize=8,
                        fontweight="bold",
                    )
                    self.ax.text(
                        col + 0.5,
                        row + 0.54,
                        "x",
                        ha="center",
                        va="center",
                        color=x_color,
                        fontsize=17,
                        fontweight="bold",
                    )
                    continue

                if snapshot.evidence.hit_mask & bit:
                    continue

                probability_label = f"{snapshot.probabilities[index] * 100:.0f}%"
                text_color = self._text_color_for_probability(snapshot.probabilities[index])
                self.ax.text(
                    col + 0.5,
                    row + 0.5,
                    probability_label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold",
                )

    def _draw_move_marker(self, snapshot: TurnSnapshot) -> None:
        row, col = divmod(snapshot.move_index, self.solver.config.cols)
        self.ax.add_patch(
            Rectangle(
                (col, row),
                1,
                1,
                fill=False,
                edgecolor="#ffffff",
                linewidth=3,
            )
        )

    def _draw_state_overlay(self, hit_matrix: np.ndarray, miss_matrix: np.ndarray) -> None:
        for row in range(hit_matrix.shape[0]):
            for col in range(hit_matrix.shape[1]):
                if hit_matrix[row, col] > 0:
                    self.ax.add_patch(
                        Rectangle(
                            (col, row),
                            1,
                            1,
                            facecolor=(0.84, 0.16, 0.16, 0.9),
                            edgecolor=(0.58, 0.06, 0.06, 1.0),
                            linewidth=1.4,
                        )
                    )
                elif miss_matrix[row, col] > 0:
                    self.ax.add_patch(
                        Rectangle(
                            (col, row),
                            1,
                            1,
                            facecolor=(0.72, 0.72, 0.72, 0.92),
                            edgecolor=(0.42, 0.42, 0.42, 1.0),
                            linewidth=1.2,
                        )
                    )

    def _text_color_for_probability(self, probability: float) -> str:
        return "white" if probability < 0.55 else "black"


def probability_map(probabilities: Sequence[float], rows: int, cols: int) -> np.ndarray:
    return np.array(probabilities, dtype=float).reshape(rows, cols)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian battleship simulation with matrix input")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="board rows")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help="board columns")
    parser.add_argument(
        "--ship",
        dest="ships_spec",
        action="append",
        default=[],
        help="fleet item in size:count format, e.g. 3:1",
    )
    parser.add_argument("--board", type=str, default=None, help="board matrix as text")
    parser.add_argument("--board-file", type=str, default=None, help="file containing the board matrix")
    parser.add_argument("--no-clear", action="store_true", help="do not clear screen between turns")
    return parser


def clear_screen(enabled: bool) -> None:
    if enabled and sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def read_board_text(args: argparse.Namespace) -> str | None:
    if args.board:
        return args.board
    if args.board_file:
        with open(args.board_file, "r", encoding="utf-8") as file_handle:
            return file_handle.read()
    return None


def prompt_for_board(config: GameConfig) -> str:
    print("Enter the board as a matrix of x and 0.")
    print(f"Expected size: {config.rows}x{config.cols}")
    print("Examples: '0 0 x 0' or '00x0'. Empty line ends input.")

    lines: list[str] = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)


def tokenize_row(line: str) -> list[str]:
    if not line:
        return []
    if any(char.isspace() for char in line):
        return [token for token in line.split() if token]
    return list(line)


def parse_matrix(text: str, expected_rows: int, expected_cols: int) -> ParsedMatrix:
    lines = [line.strip() for line in text.replace(";", "\n").splitlines() if line.strip()]
    if not lines:
        raise ValueError("board is empty")

    rows: list[list[int]] = []
    for line in lines:
        tokens = tokenize_row(line)
        if not tokens:
            continue
        row: list[int] = []
        for token in tokens:
            normalized = token.lower()
            if normalized in SHIP_TOKENS:
                row.append(1)
            elif normalized in WATER_TOKENS:
                row.append(0)
            else:
                raise ValueError(f"invalid matrix token '{token}'")
        rows.append(row)

    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(rows)}")
    for row_index, row in enumerate(rows):
        if len(row) != expected_cols:
            raise ValueError(f"expected {expected_cols} columns, got {len(row)} on row {row_index + 1}")

    return ParsedMatrix(rows=rows)


def matrix_to_placements(matrix: ParsedMatrix, solver: BayesianBattleshipSolver) -> list[Placement]:
    visited: set[tuple[int, int]] = set()
    placements: list[Placement] = []

    def neighbors(row: int, col: int) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            new_row = row + d_row
            new_col = col + d_col
            if 0 <= new_row < matrix.row_count and 0 <= new_col < matrix.col_count:
                result.append((new_row, new_col))
        return result

    for row in range(matrix.row_count):
        for col in range(matrix.col_count):
            if matrix.rows[row][col] == 0 or (row, col) in visited:
                continue

            component: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited.add((row, col))

            while queue:
                current_row, current_col = queue.popleft()
                component.append((current_row, current_col))
                for next_row, next_col in neighbors(current_row, current_col):
                    if matrix.rows[next_row][next_col] == 1 and (next_row, next_col) not in visited:
                        visited.add((next_row, next_col))
                        queue.append((next_row, next_col))

            row_set = {cell_row for cell_row, _ in component}
            col_set = {cell_col for _, cell_col in component}
            if len(row_set) != 1 and len(col_set) != 1:
                raise ValueError(f"ship at cells {component} is not straight")

            if len(row_set) == 1:
                fixed_row = next(iter(row_set))
                cols_sorted = sorted(cell_col for _, cell_col in component)
                expected = list(range(cols_sorted[0], cols_sorted[0] + len(cols_sorted)))
                if cols_sorted != expected:
                    raise ValueError(f"ship at row {fixed_row + 1} has gaps")
                indices = [fixed_row * solver.config.cols + col_index for col_index in cols_sorted]
            else:
                fixed_col = next(iter(col_set))
                rows_sorted = sorted(cell_row for cell_row, _ in component)
                expected = list(range(rows_sorted[0], rows_sorted[0] + len(rows_sorted)))
                if rows_sorted != expected:
                    raise ValueError(f"ship at column {fixed_col + 1} has gaps")
                indices = [row_index * solver.config.cols + fixed_col for row_index in rows_sorted]

            occupied_mask = 0
            forbidden_mask = 0
            for index in indices:
                occupied_mask |= solver._bit(index)
                forbidden_mask |= solver.neighbor_masks[index]

            placements.append(
                Placement(
                    size=len(indices),
                    cells=tuple(indices),
                    occupied_mask=occupied_mask,
                    forbidden_mask=forbidden_mask,
                )
            )

    return placements


def board_from_matrix_text(text: str, solver: BayesianBattleshipSolver) -> GameBoard:
    matrix = parse_matrix(text, solver.config.rows, solver.config.cols)
    ships = matrix_to_placements(matrix, solver)
    return solver.validate_board(ships)


def run_simulation(solver: BayesianBattleshipSolver, board: GameBoard, clear_between_turns: bool) -> None:
    evidence = Evidence()
    turn = 1
    snapshots: list[TurnSnapshot] = []

    while evidence.hit_mask != board.occupied_mask:
        probabilities, total_states = solver.compute_posteriors(evidence)
        move_index, move_probability = solver.choose_move(probabilities, evidence)

        snapshots.append(
            TurnSnapshot(
                turn=turn,
                probabilities=list(probabilities),
                total_states=total_states,
                move_index=move_index,
                move_probability=move_probability,
                evidence=Evidence(
                    shot_mask=evidence.shot_mask,
                    hit_mask=evidence.hit_mask,
                    miss_mask=evidence.miss_mask,
                ),
            )
        )

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

        turn += 1

    probabilities, total_states = solver.compute_posteriors(evidence)
    snapshots.append(
        TurnSnapshot(
            turn=turn - 1,
            probabilities=list(probabilities),
            total_states=total_states,
            move_index=snapshots[-1].move_index if snapshots else 0,
            move_probability=snapshots[-1].move_probability if snapshots else 0.0,
            evidence=Evidence(
                shot_mask=evidence.shot_mask,
                hit_mask=evidence.hit_mask,
                miss_mask=evidence.miss_mask,
            ),
        )
    )

    print(f"Finished in {turn - 1} turns")
    viewer = SimulationViewer(solver, board, snapshots)
    viewer.show()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = GameConfig(rows=args.rows, cols=args.cols, fleet=parse_fleet(args.ships_spec))
    solver = BayesianBattleshipSolver(config)

    try:
        board_text = read_board_text(args)
        if board_text is None:
            board_text = prompt_for_board(config)
        board = board_from_matrix_text(board_text, solver)
    except Exception as exc:
        print(f"Invalid board: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    run_simulation(solver, board, clear_between_turns=not args.no_clear)


if __name__ == "__main__":
    main()