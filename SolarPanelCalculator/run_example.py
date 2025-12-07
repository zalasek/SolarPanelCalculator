"""
Solar Panel Calculator - Example Usage
========================================
This script demonstrates how to use the SolarPanelCalculator service to compute
structural support positions (mounts) and inter-panel connection points (joints)
for a solar array installation.
"""

from src.SolarPanelCalculator.solar_calculator import SolarPanelCalculator


# X coordinate of the first rafter
rafter_x0 = 5.0

# Panel layout: 10 panels in 3 rows
example = [
  {"x": 0, "y": 0}, {"x": 45.05, "y": 0}, {"x": 90.1, "y": 0},
  {"x": 0, "y": 71.6}, {"x": 135.15, "y": 0}, {"x": 135.15, "y": 71.6},
  {"x": 0, "y": 143.2}, {"x": 45.05, "y": 143.2}, {"x": 135.15, "y": 143.2},
  {"x": 90.1, "y": 143.2}
]

# Alternative panel layout matching the example one in the task
# example = [
#     # Row 1 (y=0): 3 panels
#     {"x": 0, "y": 0},
#     {"x": 90.1, "y": 0},
#     {"x": 135.15, "y": 0},
#     # Row 2 (y=71.6): 4 panels
#     {"x": 0, "y": 71.6},
#     {"x": 45.05, "y": 71.6},
#     {"x": 135.15, "y": 71.6},
#     {"x": 90.1, "y": 71.6}
# ]

# Initialize calculator
service = SolarPanelCalculator(rafter_x0)

# Calculate mount and joint coordinates
result = service.get_coordinates(example)

# Print joint coordinates
print('Calculated joint coordinates (x,y):')
for joint in result['joint_coordinates']:
    print(joint)

# Print mount coordinates
print('\nCalculated mount coordinates (x,y):')
for mount in result['mount_coordinates']:
    print(mount)

# Plot layout
service.plot_layout(example)