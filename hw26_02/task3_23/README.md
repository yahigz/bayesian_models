The game RockPaperScissors https://en.wikipedia.org/wiki/Rock-paper-scissors is played by Player1 and Player2. The sequences of moves $x_{1:T}^1$, $x_{1:T}^2$
(each move $x_{t}^i$ being one of rock=$r$, paper=$p$, scissors=$s$) by both players is

$$\text{player1}=[r p r p r s p r s p p r r r r p r s r p p s r]$$
$$\text{player2}=[s p r s p s p s r p s r p p r r s p r s s p r]$$

Assume that player1 plays according to a first order Markov chain, defined by
$$p(x_{2:T}^1| x_{1:T-1}^2) = \prod\limits_{t=2}^Tp(x_t^1|x_{t-1}^1, x_{t-1}^2)$$
where $x_t^1 \in \{r,p,s\}$ are the moves at time $t$ by player 1 and player 2 respectively.

1. Calculate the probability given in equation for the data above.

2. How much more likely is it that player 1 is playing according to this strategy, compared to playing random moves?

3. Calculate the probability that player 1 plays each of rock, paper, scissors at the next time step t=T+1.

4. What would be the best move for player 2 to make at time t=T+1?

where T=23.