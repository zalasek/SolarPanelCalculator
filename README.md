# Solar Panel Calculator

A Python application for calculating structural support positions (mounts) and inter-panel connection points (joints) for solar array installations.

## Features

- **Rafter Calculation**: Automatically computes vertical support line positions
- **Mount Positioning**: Determines optimal attachment points for panels on rafters
- **Joint Detection**: Identifies connection points between adjacent panels
- **Layout Visualization**: Generates visual representation of the complete solar panel array
- **Structural Constraints**: Enforces engineering limits:
  - Edge clearance: minimum 2 units from panel edges
  - Cantilever limit: maximum 16 units unsupported overhang
  - Span limit: maximum 48 units between consecutive supports
  - Shared connections: joints can connect up to 4 panels in grid layouts

## Project Structure
    SolarPanelCalculator/
    ├── run_example.py
    ├── src/
    │   └── SolarPanelCalculator/
    │       └── solar_calculator.py
    ├── README.md
    └── requirements.txt
## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone or download the repository**

2. **Create a virtual environment**

   ```
    python3 -m venv venv
   ```
3. **Activate the virtual environment**

- On Windows:
  ```
  venv\Scripts\activate
  ```
- On macOS/Linux:
   ```
   source venv/bin/activate
   ```
4. **Install dependencies from requirements.txt**
   ```
   pip install -r SolarPanelCalculator/requirements.txt
   ```
### Usage
Run this command in terminal.
```
python3 run_example.py
```
This will:
  - Initialize the calculator with a rafter at x=5.0
  - Process a 10-panel solar array layout
  - Print calculated joint coordinates
  - Print calculated mount coordinates
  - Display a visual plot of the layout

### Requirements
See requirements.txt for complete dependency list.
