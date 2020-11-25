## DRL - DDPG Algorithm - Reacher Continuous Control

### Model Architecture
DDPG was used in order to solve this project along with a few modifications to
allow for 20 agent environment.

The use of two deep neural nets was the foundation behind the (Actor-Critic)
algorithm used to solve this project.

- Actor
  - 3 fully-connected Linear layers - 128 neurons in the 2 FC layers
  - reLu activation was used on the 2 hidden FC layers
  - tanH activation on final output linear layer

- Critic
  - 3 fully-connected Linear layers - 128 neurons in the 2 FC layers
  - reLu activation was used on the 2 hidden FC layers
  - linear activation fxn for final output layer

### Hyperparameters
- learning rate: 1e-3 (actor/critic DNN)
- batch size: 128
- buffer size: 1e5
- gamma: 0.99
- tau: 1e-3
- weight decay: 0
- mu: 0.0 (Ornstein-Uhlenbeck noise)
- theta: 0.15 (Ornstein-Uhlenbeck noise)
- sigma: 0.2 (Ornstein-Uhlenbeck noise)
- num episodes: 10000
- max t: 1000

## Plot of Rewards
![](./result.png)


### Future Advancements
- Add prioritized replay buffer
- A3C algorithm
- D4PG algorithm