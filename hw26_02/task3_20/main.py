if __name__ == "__main__":
    p_inc = [0.8, 0.2]
    p_w_inc = [[0.7, 0.3, 0.0, 0.0], [0.2, 0.1, 0.4, 0.3]]
    p_h_inc = [[0.2, 0.8, 0.0, 0.0], [0.0, 0.0, 0.3, 0.7]]

    p_wh_inc = [[[0.0 for _ in range(4)] for _ in range(4)] for _ in range(2)]
    for k in range(2):
        for w in range(4):
            for h in range(4):
                p_wh_inc[k][w][h] = p_w_inc[k][w] * p_h_inc[k][h]

    p_wh = [[0.0 for _ in range(4)] for _ in range(4)]
    for w in range(4):
        for h in range(4):
            p_wh[w][h] = sum(p_inc[k] * p_wh_inc[k][w][h] for k in range(2))

    print("p(w,h) table")
    print("w\\h |   1      2      3      4")
    print("----+---------------------------")
    for w in range(4):
        row = "  ".join(f"{p_wh[w][h]:.4f}" for h in range(4))
        print(f" {w + 1}  | {row}")

    p_w = [sum(p_wh[w][h] for h in range(4)) for w in range(4)]
    p_h = [sum(p_wh[w][h] for w in range(4)) for h in range(4)]

    print("\np(w):")
    for w in range(4):
        print(f"w={w + 1}: {p_w[w]:.4f}")

    print("\np(h):")
    for h in range(4):
        print(f"h={h + 1}: {p_h[h]:.4f}")

    p_wh_ind = [[p_w[w] * p_h[h] for h in range(4)] for w in range(4)]
    print("\nAssuming independence: p_ind(w,h) = p(w)p(h)")
    print("w\\h |   1      2      3      4")
    print("----+---------------------------")
    for w in range(4):
        row = "  ".join(f"{p_wh_ind[w][h]:.4f}" for h in range(4))
        print(f" {w + 1}  | {row}")

    print("As we can see, the actual p(w,h) does not match the product of p(w) and p(h), indicating that w and h are not independent.")