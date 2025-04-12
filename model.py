from environment import Environment

ages = 10
num_agents = 100
num_steps = 20
aging=True
gov=True

for age in range(ages):
    model = Environment(num_agents, aging, gov=gov)
    model.run(num_steps)

#model.plot()