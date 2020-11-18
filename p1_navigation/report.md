[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

## Implementation

### Algorithm

- Deep Q-Network

I used a vanilla Deep Q-Network to approach this problem after reading a little
about the options available. While this task is placed in a larger dimension
space the actions were still only limited to 4 choices. This made me believe I
didn't need to use a more complex algorithm to achieve good results. Therefore I
began by implementing the vanilla Q-Network architecture.

After the initial run the agent was able to achieve an avg score of 13 over 100
episodes just after 341 episodes. I wasn't sure how that compared so I looked at
the solution provided by the Udacity team. They had mentioned their agent took
less than 1800 episodes in order to converge to that point. Using that as a
benchmark I deduced this was a good starting point. Next I decided to try and
tune the hyperparameters.

### Hyperparameter Tuning

Hyperparameter tuning isn't a black and white case so I decided to start with
the architecture of the neurnal network first to see what changes could be made
there. Again after looking at the complexity of the actual problem and the
initial architecture (2 fully-connected layers each with 64 nodes) I didn't
really suspect making it much more complicated was going to be the solution.
Nonetheless I went ahead and tried a few options. Initially I increased the
number of nodes in the first fully-connected layer to 512 and left the second at
64. This did not help the training. I then tried 512 for both layers, and this
also didn't do any better. As I had suspected this was probably not where I
to make too many changes. I then decided to actually make the neural network
even simpler by reducing both fully-connected layers to only use 32 nodes.
And this resulted in a better result. I then moved on to tweak the learning
rate in both directions starting at 5e-4. After a few changes 5.5e-4 seemed
to perform the best. I lastly adjusted TAU to make the updates to the target
params more prominent moving from 1e-3 to 1e-2. After these adjustments the
agent was able to achieve an average score of 13 over 100 episodes in 35
episodes.

- Epsilon decay: eps_decay=0.0995
- Replay buffer size: BUFFER_SIZE = int(1e5)
- Minibatch size: BATCH_SIZE = 64
- Discount factor: GAMMA = 0.99
- For soft update of target parameters: TAU = 1e-2
- Learning rate: LR = 5.5e-4
- Update every: 4
- Epsilon start: 1.0
- Epsilon end: 0.01

#### Model Architecture
- 2 Linear fully connected layers each with 32 nodes
- state size: 37
- action size: 4
- ReLu activations for both FC layers

### Future Advancements

While 35 episodes is no means the quickest or most optimal solution for the
problem I feel the vanilla DQN solution does a more than reasonable job at
finding a solution. While DQN tends to overestimate I think it's less of an
issue for a task like this in comparison to say an Atari game, in which the
action space size is much larger and that overestimation can lead to bigger
biases. I do think this solution can be improved on further by both additional
hyperparameter tweaking as well as using a more complex algorithm such as the
Rainbow or Dueling Networks.