winrate_matrix_test_agents = \
"""

**Intuition**: We want a quantitative measure of transitive improvement in
performance across our test agents. That is, as our test agents receive longer
computational (training) time, they should consistently beat test agents with
lower computational budget. If the lower triangular matrix presents winrates
above 50%, it means that test agents are indeed improving monotonically.

**RL algorithm**: PPO

**Self-play algorithm**: Full history Self Play

**Agent generation**: We trained an agent using PPO for a total of $X$
Connect4 episodes, freezing a copy of the training agent every $Y$ episodes
to serve as a testing agent $\pi \in \mathbb{\pi}^{test}$ for a total of $Z$
agents. No hyperparameter sweep was conducted, and these hyperparameters were
picked intuitively.

**Matrix computation**: Winrates of every test agent against each other, where
each entry $w_{i, j}$ was computed by playing $X$ head-to-head matches between
agents $\pi^{test}_i$ and $\pi^{test}_j$. The position of each agent (first
player or second player) was chosen randomly at the beginning of each match.
"""

nash_averaging_evolutions_test_agents = \
"""

**Intuition**: Further proof of transitive performance improvement in test
agents. We want to confirm that every time a new test agent is generated and
added to the set of existing test agents, this new test agent should be "of
value". We use a notion of "value" rooted in game-theoretic aspects.

**Agents**: Test agents (generated as above)

**Figure generation**: Nash-Averaging support for all agents computed from
empirical evaluation matrix described above. The $i$th row represents the
maximum entropy Nash equilibrium associated to the Nash averaging where up the
the $i$th agent were benchmarked ($\pi^{test}_0,\ldots,\pi^{test}_i$). By
gradually including more agents we can more clearly see that whenever a new
agent was introduced, it performed better than those that came before it.
"""

mcts_equivalent_strength_test_agents = \
"""

**Intuition**: Internal metrics of performance like those defined above give a
narrow perspective of the "objective strength" of those agents. This is because
the computed test agents might be monotonically improving by learning how to
exploit very specific weaknesses of previous test agents, without distilling
generalized information as to how to play the game. To measure this
"generalized information for how to play the game" we can use MCTS.

**MCTS equivalent strength computation**: For each test agent $\pi^{test}_i \in
\pi^{test}$, we match it against an MCTS($B$) agent with a low computational
budget $B$ for $X$ episodes ($X / 2$ on each position). We slowly increase the
computational budget $B$ until the MCTS agent obtains a $50\%$ winrate against
agent $\pi^{test}_i$. We denote this approximated budget $B_i$ as the **MCTS
equivalent strength** of the test agent $\pi^{test}_i$.

**Plot**: TODO: We have a list of test agents and their corresponding MCTS
equivalent strength $(\pi^{test}_i, B_i)$ for all $\pi^{test}_i \in \pi^{test}$.

"""


winrate_matrix_mcts_agents = \
"""

**Intuition**: The test agents show transitive improvement. We want to observe
if their MCTS($B_i$) equivalent agent also follow this transitive fashion.

**Matrix computation**: Winrates of every MCTS equivalent against each other,
where each entry $w_{i, j}$ was computed by playing $X$ head-to-head matches
between agents MCTS($B_i$) and MCTS($B_j$). The position of each agent (first
player or second player) was chosen randomly at the beginning of each match.
"""

winrate_matrix_mcts_and_test_agents = \
"""

**Intuition**: TODO

**Matrix computation**: Like the previous winrate matrix computations, but
using a combined population of both the test agents and their MCTS equivalent
agents.

"""

nash_averaging_mcts_and_test_agents = \
"""

**Intuition**: We would like to see that for a test agent, the corresponding
MCTS equivalent agent features the same support under Nash-Averaging.
"""
