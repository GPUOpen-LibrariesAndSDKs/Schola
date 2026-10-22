# Faster actor for demo recordings

Actor Adam learning rate: **0.006** (previously 0.003).
Critic rate: 0.003. Entropy weight: 0.4. All other learning and game rules unchanged.

`native-tests.txt` and `native-learning-curve.csv` retain the four-seed test.
All seeds learned both runner preferences across a bias reversal. First-round
probability shifts were approximately doubled (0.66–1.14 percentage points,
previously 0.33–0.57); subsequent shifts depend on rewards and optimizer state.
Doubling the rate does not guarantee halving the rounds needed to learn.

A preliminary actor rate of 0.01 was rejected because seed 0 failed to reverse
its preference. The retained results use 0.006.

Editor and packaged builds succeeded. `arena-packaged.json` and
`arena-packaged.log` retain all nine passing integration checks at the new rate:
512 experiences, 512 updates, actor-only action choices, P(LEFT) 0.927461 after
the left-biased phase and 0.017293 after the right-biased phase without resetting
weights. Recent catch rate at the end was 0.75. The scored transitions are saved
in `arena-packaged-experiences.csv`.
