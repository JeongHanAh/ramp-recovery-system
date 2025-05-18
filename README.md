# Highway Ramp Path Recovery System

## Project Overview
This system is designed to correct distortions in highway ramp trajectories using a sophisticated path recovery algorithm. It implements a B-spline based curve modeling approach combined with real-time correction vectors to ensure smooth and accurate path recovery.

## Key Features
- B-spline curve modeling for reference path generation
- Real-time path correction using tangential and error vectors
- Curvature-based ramp section identification
- Visualization of path correction process with animations
- GPS/IMU sensor data integration capability

## Project Structure
```
ramp-recovery-system/
├── src/
│   ├── data_processing/
│   │   ├── ramp_data_processor.py  # Ramp data processing and curve extraction
│   │   └── sensor_data.py          # GPS/IMU sensor data processing
│   ├── curve_modeling/
│   │   └── spline_curve.py         # B-spline curve fitting and derivatives
│   ├── path_recovery/
│   │   └── recovery_algo.py        # Path correction algorithm
│   └── visualization/
│       └── ramp_visualizer.py      # Data visualization and animation
├── data/
│   └── raw/
│       └── reference_paths/        # Reference path data storage
└── results/                        # Output visualization storage
```

## Technical Details

### Core Components

1. **Ramp Data Processor (`ramp_data_processor.py`)**
   - Extracts ramp sections using curvature analysis
   - Applies Savitzky-Golay filtering for noise reduction
   - Handles coordinate transformations

2. **Path Recovery Algorithm (`recovery_algo.py`)**
   - Implements correction vector calculation
   - Combines tangential and error vectors
   - Updates position using adaptive step sizes

3. **B-spline Modeling (`spline_curve.py`)**
   - Fits B-spline curves to reference paths
   - Computes curve derivatives for tangent vectors
   - Provides smooth path interpolation

4. **Visualization System (`ramp_visualizer.py`)**
   - Real-time visualization of path correction
   - Animated correction process
   - Comparison between original and corrected paths

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ramp-recovery-system.git
cd ramp-recovery-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your reference path data in JSON format:
```json
{
    "coordinates": {
        "x": [...],
        "y": [...]
    }
}
```

2. Run the main program:
```bash
python src/main.py
```

3. Check the results in the `results/` directory:
   - `ramp_analysis.png`: Static visualization
   - `correction_animation.gif`: Animation of the correction process

## Dependencies
- numpy
- scipy
- matplotlib
- pyproj (for GPS coordinate transformations)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
