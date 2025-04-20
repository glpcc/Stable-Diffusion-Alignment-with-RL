import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def value_to_color(value, vmin=80, vmax=200, cmap_name='viridis'):
    # Normalize value between 0 and 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm(value))
    
    # Convert RGBA to HEX
    hex_color = mcolors.to_hex(rgba)
    return hex_color

# Example: Create colors for a list of values
values = [86, 164, 130, 180, 123]
colors = [value_to_color(v) for v in values]

# Print value-color pairs
for v, c in zip(values, colors):
    print(f"Value: {v} -> Color: {c}")
