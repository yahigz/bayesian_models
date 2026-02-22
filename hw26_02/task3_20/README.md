A survey of households in which the husband and wife each own their own car is made. The survey also states whether each household income (inc) is high or low. There are 4 car types, $dom(h) = dom(w) = \{1,2,3,4\}$, the first two being cheap and the last two being expensive. The survey finds that, given their household income, the types of cars owned by a husband and wife are independent:
$$
\text{wife's car type} ⊥⊥ \text{husband's car type} | \text{family income}
$$
Specifically, $p(inc = low) = 0.8$ and

$$
p(w|inc = low) = \begin{pmatrix} 0.7 \\ 0.3 \\ 0 \\ 0 \end{pmatrix}, \quad 
p(w|inc = high) = \begin{pmatrix} 0.2 \\ 0.1 \\ 0.4 \\ 0.3 \end{pmatrix}
$$
$$
p(h|inc = low) = \begin{pmatrix} 0.2 \\ 0.8 \\ 0 \\ 0 \end{pmatrix}, \quad 
p(h|inc = high) = \begin{pmatrix} 0 \\ 0 \\ 0.3 \\ 0.7 \end{pmatrix}
$$
Find the marginal $p(w,h)$ and show that whilst $h ⊥⊥ w | inc$, it is not the case that $h ⊥⊥ w$.