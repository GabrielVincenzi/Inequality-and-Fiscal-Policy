from environment import Environment
import random

ages = 1
num_agents = 10
num_steps = 10
aging=True
gov=True
taxSchedule=False

for age in range(ages):
    #num_steps = random.choice([20, 30])
    model = Environment(num_agents, aging, gov=gov, taxSchedule=taxSchedule)
    model.run(num_steps)

df = model.get_data()
df.to_csv('modelData', sep=";")
model.plot()