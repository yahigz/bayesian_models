from collections import defaultdict



if __name__ == "__main__":
    p1_seq = "r p r p r s p r s p p r r r r p r s r p p s r".split()
    p2_seq = "s p r s p s p s r p s r p p r r s p r s s p r".split()

    T = len(p1_seq)
    moves = ["r", "p", "s"]

    counts = defaultdict(lambda: {m: 0 for m in moves})
    for t in range(1, T):
        prev = (p1_seq[t - 1], p2_seq[t - 1])
        counts[prev][p1_seq[t]] += 1

    p = 1.0
    for t in range(1, T):
        prev = (p1_seq[t - 1], p2_seq[t - 1])
        total = sum(counts[prev].values())
        prob = counts[prev][p1_seq[t]] / total
        p *= prob


    random = (1.0 / 3.0) **(T - 1)
    ratio = p / random

    print("1) p(x_2:T^1 | x_1:T-1^2) under empirical model:")
    print(f"   p     = {p:.6e}")

    print("\n2) Likelihood ratio vs random moves:")
    print(f"   ratio     = {ratio:.6e}")

    last_prev = (p1_seq[-1], p2_seq[-1])
    if last_prev in counts and sum(counts[last_prev].values()) > 0:
        total = sum(counts[last_prev].values())
        p1_next = {m: counts[last_prev][m] / total for m in moves}
    else:
        p1_next = {m: 1.0 / 3.0 for m in moves}

    print("\n3) p1 next-move distribution at t=T+1:")
    for m in moves:
        print(f"   p(x_{T + 1}^1={m}) = {p1_next[m]:.4f}")

    best_move = "undefined"
    best_prob = 0.0
    for m in moves:
        if p1_next[m] > best_prob:
            best_prob = p1_next[m]
            if m == "r":
                best_move = "p"
            elif m == "p":
                best_move = "s"
            elif m == "s":
                best_move = "r"
    print("\n4) Best move for player 2 at t=T+1:")
    print(f"   best move: {best_move}")
