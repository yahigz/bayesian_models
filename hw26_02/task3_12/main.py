

def GetUndirectedGraph(A):
    n = len(A)
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if A[i][j] == 1 or A[j][i] == 1:
                G[i][j] = 1
                G[j][i] = 1
    return G

def GetImmoralities(A):
    n = len(A)
    immoralities = set()
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                if A[i][j] == 1 and A[k][j] == 1 and A[i][k] == 0 and A[k][i] == 0:
                    immoralities.add((min(i, k), max(i, k), j))
    return immoralities

def MarkovEquiv(A, B):
    skeleton_A = GetUndirectedGraph(A)
    skeleton_B = GetUndirectedGraph(B)
    if skeleton_A != skeleton_B:
        return False
    immoralities_A = GetImmoralities(A)
    immoralities_B = GetImmoralities(B)
    return immoralities_A == immoralities_B

if __name__ == "__main__":
    A = [
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ]
    B = [
        [0, 0, 0],
        [1, 0, 1],
        [0, 0, 0]
    ]
    print(MarkovEquiv(A, B))