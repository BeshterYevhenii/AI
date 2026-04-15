import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

temp = ctrl.Antecedent(np.arange(0, 41, 1), 'temp')

temp_rate = ctrl.Antecedent(np.arange(-5, 6, 0.1), 'temp_rate')

ac_dial = ctrl.Consequent(np.arange(-90, 91, 1), 'ac_dial')

temp['very_cold'] = fuzz.trimf(temp.universe, [0, 0, 15])
temp['cold'] = fuzz.trimf(temp.universe, [10, 18, 22])
temp['normal'] = fuzz.trimf(temp.universe, [20, 24, 28])
temp['warm'] = fuzz.trimf(temp.universe, [25, 30, 35])
temp['very_warm'] = fuzz.trimf(temp.universe, [32, 40, 40])

temp_rate['negative'] = fuzz.trimf(temp_rate.universe, [-5, -5, 0])
temp_rate['zero'] = fuzz.trimf(temp_rate.universe, [-1, 0, 1])
temp_rate['positive'] = fuzz.trimf(temp_rate.universe, [0, 5, 5])

ac_dial['large_left'] = fuzz.trimf(ac_dial.universe, [-90, -90, -45])
ac_dial['small_left'] = fuzz.trimf(ac_dial.universe, [-60, -30, 0])
ac_dial['off'] = fuzz.trimf(ac_dial.universe, [-10, 0, 10])
ac_dial['small_right'] = fuzz.trimf(ac_dial.universe, [0, 30, 60])
ac_dial['large_right'] = fuzz.trimf(ac_dial.universe, [45, 90, 90])

rules = [
    ctrl.Rule(temp['very_warm'] & temp_rate['positive'], ac_dial['large_left']),      
    ctrl.Rule(temp['very_warm'] & temp_rate['negative'], ac_dial['small_left']),      
    ctrl.Rule(temp['warm'] & temp_rate['positive'], ac_dial['large_left']),           
    ctrl.Rule(temp['warm'] & temp_rate['negative'], ac_dial['off']),                  
    ctrl.Rule(temp['very_cold'] & temp_rate['negative'], ac_dial['large_right']),     
    ctrl.Rule(temp['very_cold'] & temp_rate['positive'], ac_dial['small_right']),     
    ctrl.Rule(temp['cold'] & temp_rate['negative'], ac_dial['large_right']),          
    ctrl.Rule(temp['cold'] & temp_rate['positive'], ac_dial['off']),                  
    ctrl.Rule(temp['very_warm'] & temp_rate['zero'], ac_dial['large_left']),          
    ctrl.Rule(temp['warm'] & temp_rate['zero'], ac_dial['small_left']),               
    ctrl.Rule(temp['very_cold'] & temp_rate['zero'], ac_dial['large_right']),         
    ctrl.Rule(temp['cold'] & temp_rate['zero'], ac_dial['small_right']),              
    ctrl.Rule(temp['normal'] & temp_rate['positive'], ac_dial['small_left']),         
    ctrl.Rule(temp['normal'] & temp_rate['negative'], ac_dial['small_right']),        
    ctrl.Rule(temp['normal'] & temp_rate['zero'], ac_dial['off'])                     
]

ac_ctrl = ctrl.ControlSystem(rules)
ac_sim = ctrl.ControlSystemSimulation(ac_ctrl)

current_temp = 35         
current_temp_rate = 2.0   

ac_sim.input['temp'] = current_temp
ac_sim.input['temp_rate'] = current_temp_rate

ac_sim.compute()

print("\nЗадача 2: Керування кондиціонером")
print(f"Поточна температура: {current_temp} °C")
print(f"Швидкість зміни: {current_temp_rate} °C/хв")
print(f"Поворот регулятора кондиціонера: {ac_sim.output['ac_dial']:.2f} градусів")

if ac_sim.output['ac_dial'] < -10:
    print("Статус: Включено режим ХОЛОД (поворот вліво)")
elif ac_sim.output['ac_dial'] > 10:
    print("Статус: Включено режим ТЕПЛО (поворот вправо)")
else:
    print("Статус: ВИМКНЕНО")

ac_dial.view(sim=ac_sim)
plt.title("Регулятор кондиціонера")
plt.show()
