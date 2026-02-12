import os
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt


sun_df = pd.read_csv('input.csv')
sun_df = sun_df.rename(columns = {'length': 'Wavelength', 'irradiance': 'Spectral irradiance', 'cumulative_photon_flux': 'Cumulative photon flux'})
sun_df['Wavelength'] = sun_df['Wavelength'].apply(lambda x: x * 10 ** 9)

folder_path = 'lamps'
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
all_lamps = []
for file in all_files:
    file_path = os.path.join(folder_path, file)
    lamp_name = os.path.splitext(file)[0]
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        'lenght': 'Wavelength',
        'intensive': 'Spectral irradiance',
        'cumulative': 'Cumulative photon flux'
    })
    df.insert(0, 'Lamp', lamp_name)
    df = df.apply(pd.to_numeric, errors = 'coerce').dropna()
    all_lamps.append(df)
lamps_df = pd.concat(all_lamps, ignore_index = True)


wavelengths = sun_df['Wavelength'].values
features = ['Spectral irradiance', 'Cumulative photon flux']
lamps_types = lamps_df['Lamp'].unique()

b = []
for wl in wavelengths:
    row = sun_df[sun_df['Wavelength'] == wl]
    for feat in features:
        b.append(row[feat].values[0])
b = np.array(b)

A = []
for lamp in lamps_types:
    lamp_data = lamps_df[lamps_df['Lamp'] == lamp]
    lamp_features = []
    for wl in wavelengths:
        row = lamp_data[lamp_data['Wavelength'] == wl]
        for feat in features:
            if not row.empty:
                lamp_features.append(row[feat].values[0])
            else:
                lamp_features.append(0.0)
    A.append(lamp_features)
A = np.array(A).T


lambda_candidates = np.logspace(-4, 1, 30)

best_solution = None
best_lambda = None
min_nonzero = np.inf

for lambda_reg in lambda_candidates:
    x = cp.Variable(A.shape[1], nonneg = True)
    # Лямбда - коэффициент регуляризации, который управляет балансом между точностью и разреженностью
    objective = cp.Minimize(cp.sum_squares(A @ x - b) + lambda_reg * cp.norm1(x))
    problem = cp.Problem(objective)
    problem.solve()

    if x.value is None:
        continue

    x_val = x.value
    # Относительная ошибка в процентах от эталона
    rel_error = np.linalg.norm(A @ x_val - b) / np.linalg.norm(b)
    nonzero_count = np.sum(x_val > 1e-6)

    if rel_error <= 0.15 and nonzero_count < min_nonzero:
        best_solution = x_val
        best_lambda = lambda_reg
        min_nonzero = nonzero_count


if best_solution is not None:
    results = pd.DataFrame({
        'Lamp': lamps_types,
        'Coefficient': best_solution
    })
    results = results[results['Coefficient'] > 1e-6].reset_index(drop = True)
    print(results)
else:
    print('\nНе найдено решение с ошибкой ≤ 15%')


reconstructed = A @ best_solution
wavelengths = sun_df['Wavelength'].values
features = ['Spectral irradiance', 'Cumulative photon flux']

fig, axes = plt.subplots(2, 1, figsize = (8, 6), sharex = True)

for i, feat in enumerate(features):
    y_true = [sun_df.loc[sun_df['Wavelength'] == wl, feat].values[0] for wl in wavelengths]
    y_pred = reconstructed[i::len(features)]

    axes[i].plot(wavelengths, y_true, 'o-', label = 'Солнце')
    axes[i].plot(wavelengths, y_pred, 's--', label = 'Комбинация ламп')
    axes[i].set_ylabel(feat)
    axes[i].legend()
    axes[i].grid(True)

axes[1].set_xlabel('Длина волны (нм)')
plt.tight_layout()
plt.show()
