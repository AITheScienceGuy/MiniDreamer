# MiniDreamer
An attempt at implementing a Dreamer-style agent on space invaders using a RSSM model.

## World Model
The world model is a latent dynamics model that creates useful latents for reconstruction. There are done heads and rewards heads that are useful for the actor-critic training. There is an encoder that encodes the first frame of the environment (pixel space to latent space) and then from there the latent dynamics model takes a latent and action as input and outputs the next latent and is optimized for the frame reconstruction task.

## Actor-Critic
The actor critic takes a latent as input and outputs an action for the agent to take. After the first frame is encoded into a latent, the actor takes this as input and outputs an action. This actions is given to the world model which then predicts whether or not the environment should termninate, the expected return from the agents action, and the next latent state of the environment.

## Buffer
The buffer is what holds the experiences that the world model is trained on. When we first initialize training, a random agent is used to collect experiences. The world model then trains on these experiences and the agent is trained on the latents produced by this world model. For the next iteration of training, the buffer is filled with more experiences collected by the agent trained in the past iteration. The idea is that as the agent gets better, the experiences it collects will be from later environment states that it has not already seen which would be useful for both the world model and actor critic training. We then train the iteration on the most recent percentage of the buffer as this is the most useful information to train on. The training then continues in these iterative loops.  

## Evaluation 
Both play.py and benchmark.py are used to evaluate the agent. Play lets us watch the agent playing in a real environment and benchmark runs the agent on the environment 50 times and sees its average performance to decide whether it is better than some baseline.

## Results and Future Work
Reinforcement learning takes several million environment steps on space invaders in order to learn a good policy. I tried to train this on my laptop's RTX 3050 and it was painfully slow, but I did notice world model reconstruction loss steadily trending downwards and agent reward return trending upwards. I then rented an RTX 5090 to train on but this would still have been a multi-day training run which I do not have the funds for right now so training was abandoned but I am confident that the aim of the project would be achieved.

In the future I would like to train the model fully for closure. I have also seen some work in the literature on imagination learning but with a decoder-free world model. The latents produced by the world model can more easily compress task specific information rather than information used for pixel space reconstruction. I would like to experiment with such a model and then, once the world model and agent has been trained, train a decoder from the latents to pixel space to see if it is possible to interpret what the agent is planning.

