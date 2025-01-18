#!/bin/bash

if [ -d "venv" ]; then
    source venv/Scripts/activate
else
    python3.8 -m venv venv
    source venv/Scripts/activate        
fi

# install dependencies
pip install -r requirements.txt

# Train
python train-scripts/model-train.py

# Run app
python app.py

# give this file excecutable permissions
# =================================================================
# chmod +x run_pipeline.sh
#
# Now you can run the pipeline using the command:
#
# ./run_pipeline.sh
#
# This will create a virtual environment, install the required dependencies,
# train the model, and start the Flask application.
#