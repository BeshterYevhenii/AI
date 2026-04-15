import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
pressure = ctrl.Antecedent(np.arange(0, 101, 1), 'pressure')

hot_valve = ctrl.Consequent(np.arange(-90, 91, 1), 'hot_valve')
cold_valve = ctrl.Consequent(np.arange(-90, 91, 1), 'cold_valve')

temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 25])
temperature['cool'] = fuzz.trimf(temperature.universe, [15, 30, 45])
temperature['warm'] = fuzz.trimf(temperature.universe, [35, 50, 65])
temperature['not_very_hot'] = fuzz.trimf(temperature.universe, [55, 75, 90])
temperature['hot'] = fuzz.trimf(temperature.universe, [80, 100, 100])

pressure['weak'] = fuzz.trimf(pressure.universe, [0, 0, 40])
pressure['not_very_strong'] = fuzz.trimf(pressure.universe, [30, 50, 70])
pressure['strong'] = fuzz.trimf(pressure.universe, [60, 100, 100])

for valve in [hot_valve, cold_valve]:
    valve['large_left'] = fuzz.trimf(valve.universe, [-90, -90, -60])
    valve['medium_left'] = fuzz.trimf(valve.universe, [-75, -45, -15])
    valve['small_left'] = fuzz.trimf(valve.universe, [-30, -15, 0])
    valve['zero'] = fuzz.trimf(valve.universe, [-10, 0, 10])
    valve['small_right'] = fuzz.trimf(valve.universe, [0, 15, 30])
    valve['medium_right'] = fuzz.trimf(valve.universe, [15, 45, 75])
    valve['large_right'] = fuzz.trimf(valve.universe, [60, 90, 90])

rules = [
    ctrl.Rule(temperature['hot'] & pressure['strong'], [hot_valve['medium_left'], cold_valve['medium_right']]),
    ctrl.Rule(temperature['hot'] & pressure['not_very_strong'], [hot_valve['zero'], cold_valve['medium_right']]),
    ctrl.Rule(temperature['not_very_hot'] & pressure['strong'], [hot_valve['small_left'], cold_valve['zero']]),
    ctrl.Rule(temperature['not_very_hot'] & pressure['weak'], [hot_valve['small_right'], cold_valve['small_right']]),
    ctrl.Rule(temperature['warm'] & pressure['not_very_strong'], [hot_valve['zero'], cold_valve['zero']]),
    ctrl.Rule(temperature['cool'] & pressure['strong'], [hot_valve['medium_right'], cold_valve['medium_left']]),
    ctrl.Rule(temperature['cool'] & pressure['not_very_strong'], [hot_valve['medium_right'], cold_valve['small_left']]),
    ctrl.Rule(temperature['cold'] & pressure['weak'], [hot_valve['large_right'], cold_valve['zero']]),
    ctrl.Rule(temperature['cold'] & pressure['strong'], [hot_valve['medium_left'], cold_valve['medium_right']]),
    ctrl.Rule(temperature['warm'] & pressure['strong'], [hot_valve['small_left'], cold_valve['small_left']]),
    ctrl.Rule(temperature['warm'] & pressure['weak'], [hot_valve['small_right'], cold_valve['small_right']])
]

water_ctrl = ctrl.ControlSystem(rules)
water_sim = ctrl.ControlSystemSimulation(water_ctrl)

water_sim.input['temperature'] = 85
water_sim.input['pressure'] = 80

water_sim.compute()

print("Задача 1: Керування кранами")
print(f"Поворот гарячого крану: {water_sim.output['hot_valve']:.2f} градусів")
print(f"Поворот холодного крану: {water_sim.output['cold_valve']:.2f} градусів")

hot_valve.view(sim=water_sim)
plt.title("Кран гарячої води")
plt.show()