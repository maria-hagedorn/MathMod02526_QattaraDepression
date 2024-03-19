import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def displacement_vector(x_1, x_2, include_height=False):

    def longitude_to_distance_factor(degrees):
        phi = np.deg2rad(degrees)
        return 111132.92 - 559.82 * np.cos(2 * phi) + 1.175 * np.cos(4 * phi) - 0.0023 * np.cos(6 * phi)

    def latitude_to_distance_factor(degrees):
        phi = np.deg2rad(degrees)
        return 111412.84 * np.cos(phi) - 93.5 * np.cos(3 * phi) + 0.118 * np.cos(5 * phi)

    delta_latitude = abs(x_1['latitude'] - x_2['latitude'])
    delta_longitude = abs(x_1['longitude'] - x_2['longitude'])
    mean_latitude = min(x_1['latitude'], x_2['latitude']) + delta_latitude / 2
    mean_longitude = min(x_1['longitude'], x_2['longitude']) + delta_longitude / 2

    delta_x = longitude_to_distance_factor(mean_latitude) * delta_longitude
    delta_y = latitude_to_distance_factor(mean_longitude) * delta_latitude
    delta_h = abs(x_1['h'] - x_2['h'])

    distance = [delta_x, delta_y, int(include_height)*delta_h]

    return distance


def fixed_distance_polygonal_chain_interpolation(points, distance):

    r = distance

    def k(x0, x1, x2, r):

        a = np.linalg.norm(x2-x1)**2
        b = 2*np.dot(x2-x1, x1-x0)
        c = np.linalg.norm(x1-x0)**2-r**2

        return (-b + np.sqrt(b**2-4*a*c))/(2*a)

    interpolated_points = [points[0]]
    index = 0
    indices = []

    while True:

        indices.append(index)  # used to interpolate z values (since we only want the distance between points to be constant in the (x,y)-plane
        added_point = False  # Flag to check if a point was added in this iteration

        for j, next_point in enumerate(points[index:], index):
            if np.linalg.norm(interpolated_points[-1] - next_point) >= r:
                index = j
                x0, x1, x2 = interpolated_points[-1], points[j - 1], points[j]
                x_star = x1 + k(x0, x1, x2, r) * (x2 - x1)
                interpolated_points.append(x_star)
                added_point = True
                break

        if not added_point:
            break

    return interpolated_points, indices


df = pd.read_csv("channel_data.csv", sep=",")

df['distance'] = df.apply(lambda row: np.linalg.norm(displacement_vector(row, df.iloc[0, :], include_height=False)), axis=1)
df['delta_distance'] = df['distance'] - df['distance'].shift(1)
df['x'] = df.apply(lambda row: displacement_vector(row, df.iloc[0, :], include_height=False)[0], axis=1)
df['y'] = df.apply(lambda row: displacement_vector(row, df.iloc[0, :], include_height=False)[1], axis=1)

points = df[['x', 'y']].to_numpy()

interpolated_points, indices = fixed_distance_polygonal_chain_interpolation(points, distance=250)


# Here we interpolate the z values.
# Remember only the (x,y) distance is constant 'r'. The (x,y,z) distance can be greater.
# We need to find the z value for the interpolated (x,y) points.
# We do this by defining a line between the points neighbouring the interpolated point.

z_values = []


def get_z_value(a, b, p):
    return a[2] + (b[2]-a[2])*(p[1]-a[1])/(b[1]-a[1])


for i, j in enumerate(indices):
    a = df[['x', 'y', 'h']].to_numpy()[j-1]
    b = df[['x', 'y', 'h']].to_numpy()[j]
    p = interpolated_points[i]
    z_values.append(get_z_value(a, b, p))

interpolated_points_3d = [(x, y, z) for (x, y), z in zip(interpolated_points, z_values)]


x_coords, y_coords, z_coords = zip(*interpolated_points_3d)
fig = plt.figure(figsize=(20, 12), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_coords, y_coords, z_coords, marker='.', color="red")
ax.plot(df['x'], df['y'], df['h'], marker=',', color="blue")
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$h$')
import matplotlib.ticker as ticker
ax.set_box_aspect(aspect = ((max(df['x'])-min(df['x']))/(max(df['y'])-min(df['y'])), 1, 1/5))
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=2))
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
ax.plot(y_coords, z_coords, marker='.', color="red", label='Interpolated Points')
ax.plot(df['y'], df['h'], marker=',', color="blue", label='Original Data')
ax.axhline(0, color='black', linewidth=1, linestyle='--', label='Sea Level')
ax.set_xlabel('$y$')
ax.set_ylabel('$h$')
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.tight_layout()
ax.legend()
plt.show()