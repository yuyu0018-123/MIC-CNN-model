import numpy as np
import pandas as pd
# By consulting publicly available technical materials such as the Japan Railway Technology Research Institute (JRTR) and Shinkansen design reports, a reasonable range of training set evaluation indicators has been set.
# 1. Slender ratio (I1:4-25): Referring to the design experience of the E5/E6 series of Shinkansen trains with a front slenderness ratio between 8-18
# 2. Head cross-sectional area (I2: 13-15m ²): Based on the constraints of train cross-sectional dimensions and aerodynamic resistance optimization
# 3. Cross sectional area change rate (I3:1-2.5): reflects the smooth transition degree of the front of the car and affects the generation of eddy currents
# 4. Longitudinal section shape factor (I4:1-2): controls the curvature of the vertical contour line of the front of the car
# 5. Cross section shape factor (I5:1-2): controls the curvature of the horizontal contour line of the front of the car
# 6. Resistance coefficient (I6:1-2): a key indicator that comprehensively reflects aerodynamic performance
# 7. Nose cone taper (I7:1-2): affects the micro pressure wave effect during tunnel passage
def generate_train_head_samples(n_samples=80):
    np.random.seed(42)
    data = {'I1': [], 'I2': [], 'I3': [], 'I4': [], 'I5': [], 'I6': [], 'I7': []}
    
    for i in range(n_samples):
        if i < n_samples * 0.3:
            i1 = np.random.uniform(4.0, 10.0)
        elif i < n_samples * 0.7:
            i1 = np.random.uniform(10.0, 18.0)
        else:
            i1 = np.random.uniform(18.0, 25.0)
        
        base_i2 = 13.5 + 0.055 * i1 #The cross-sectional area increases moderately with the increase of slender ratio
        i2 = base_i2 + np.random.normal(0, 0.05)
        base_i3 = 1.2 + 0.037 * i1 #The area change rate is positively correlated with the slenderness ratio
        i3 = base_i3 + np.random.normal(0, 0.03)
        base_i4 = 1.0 + 0.040 * i1 #The longitudinal section shape tends to become slender and elongated
        i4 = base_i4 + np.random.normal(0, 0.03)
        base_i5 = 1.8 - 0.035 * i1 #Optimization of cross-sectional shape with increasing slenderness ratio
        i5 = base_i5 + np.random.normal(0, 0.03)
        base_i6 = 2.0 - 0.040 * i1 #The drag coefficient decreases as the slenderness ratio increases
        i6 = base_i6 + np.random.normal(0, 0.03)
        base_i7 = 2.0 - 0.045 * i1 #The taper of the nasal cone decreases with increasing speed
        i7 = base_i7 + np.random.normal(0, 0.03)
        
        data['I1'].append(np.clip(i1, 4.0, 25.0))
        data['I2'].append(np.clip(i2, 13.0, 15.0))
        data['I3'].append(np.clip(i3, 1.0, 2.5))
        data['I4'].append(np.clip(i4, 1.0, 2.0))
        data['I5'].append(np.clip(i5, 1.0, 2.0))
        data['I6'].append(np.clip(i6, 1.0, 2.0))
        data['I7'].append(np.clip(i7, 1.0, 2.0))
    
    df = pd.DataFrame(data)
#Simulation data of train models based on publicly available technical information
    def calculate_f_value(row):
        x1 = row['I1']
        x2 = row['I2']
        x3 = row['I3']
        x4 = row['I4']
        x5 = row['I5']
        x6 = row['I6']
        x7 = row['I7']
        
        f = (
            0.85 +
            0.038 * x1 +
            0.0005 * x1**2 -
            0.001 * x1 * x6 +
            0.002 * (x2 - 14.0) -
            0.008 * (x6 - 1.5) +
            0.004 * (x4 - 1.5) +
            0.001 * (x3 - 1.5) -
            0.002 * (x5 - 1.3) +
            0.001 * (x7 - 1.5) +
            0.0001 * np.sin(x1 * 0.5)
        )
        
        f = np.clip(f, 0.9, 2.0)
        return round(f, 3)
    
    df['F_value'] = df.apply(calculate_f_value, axis=1)
    return df