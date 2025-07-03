import numpy as np
import matplotlib.pyplot as plt

from blast.utils.demo import generate_sample_data

simulation_inputs = generate_sample_data()

fig, ax1 = plt.subplots()
ax1.plot(simulation_inputs['Time_s'][:25] / 3600, simulation_inputs['SOC'][:25], '-k')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('State-of-charge')

ax2 = ax1.twinx()
ax2.plot(simulation_inputs['Time_s'][:25] / 3600, simulation_inputs['Temperature_C'][:25], '-r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylabel('Temperature (Celsius)', color='r')
plt.show()

plt.plot(simulation_inputs['Time_s'] / (3600*24*365), simulation_inputs['Temperature_C'])
plt.xlabel('Time (years)')
plt.ylabel('Temperature (Celsius)')
plt.show()

