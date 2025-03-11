from nicegui import ui
import subprocess
import sys
import os
from pathlib import Path
import threading
import time

# Function to run the script with the selected parameters
def run_primme_simulation(parameters, console_output):
    # Build command with parameters
    cmd = [sys.executable, "run_script.py"]
    for key, value in parameters.items():
        if value is not None and value != "":
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.append(f"--{key}={value}")
    
    # Create process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Read and display output in real-time
    for line in iter(process.stdout.readline, ''):
        console_output.push(line)
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code == 0:
        console_output.push("Process completed successfully!")
    else:
        console_output.push(f"Process failed with return code {return_code}")

# Main application
def create_app():
    # Initialize parameters with defaults
    parameters = {
        "trainset": "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5",
        "modelname": None,
        "dims": 2,
        "if_plot": False,
        "num_eps": 1000,
        "obs_dim": 17,
        "act_dim": 17,
        "lr": 5e-5,
        "reg": 1,
        "nsteps": 1000,
        "n_samples": 200,
        "mode": "Single_Step",
        "grain_shape": "grain",
        "grain_size": 512,
        "voroni_loaded": False,
        "ic": "./data/ic.npy",
        "ea": "./data/ea.npy",
        "ma": "./data/ma.npy",
        "ic_shape": "grain(512_512_512)",
        "size": 93,
        "dimension": 3,
        "ngrain": 2**14,
        "primme": None,
        "pad_mode": "circular"
    }
    
    # Create UI
    # Use ui.colors instead of dark_mode if that's causing issues
    ui.colors(primary='#1976D2')
    
    with ui.tabs().classes('w-full') as tabs:
        model_tab = ui.tab('Model Parameters')
        grain_tab = ui.tab('Grain Parameters')
        run_tab = ui.tab('Run Simulation')
        results_tab = ui.tab('Results')
    
    with ui.tab_panels(tabs, value=model_tab).classes('w-full'):
        with ui.tab_panel(model_tab):
            with ui.card().classes('w-full'):
                ui.label('PRIMME Model Parameters').classes('text-xl font-bold')
                
                ui.input(label='Training Set', value=parameters['trainset'], 
                         on_change=lambda e: parameters.update({"trainset": e.value})).classes('w-full')
                
                ui.input(label='Model Name (leave empty to train new model)', value=parameters['modelname'] or "", 
                         on_change=lambda e: parameters.update({"modelname": e.value if e.value else None})).classes('w-full')
                
                with ui.row():
                    ui.number(label='Dimensions', value=parameters['dims'], 
                             on_change=lambda e: parameters.update({"dims": int(e.value)}))
                    ui.number(label='Observation Dimension', value=parameters['obs_dim'], 
                             on_change=lambda e: parameters.update({"obs_dim": int(e.value)}))
                    ui.number(label='Action Dimension', value=parameters['act_dim'], 
                             on_change=lambda e: parameters.update({"act_dim": int(e.value)}))
                
                with ui.row():
                    ui.number(label='Learning Rate', value=parameters['lr'], format='%.5f', 
                             on_change=lambda e: parameters.update({"lr": float(e.value)}))
                    ui.number(label='Regularization', value=parameters['reg'], 
                             on_change=lambda e: parameters.update({"reg": float(e.value)}))
                
                with ui.row():
                    ui.number(label='Number of Epochs', value=parameters['num_eps'], 
                             on_change=lambda e: parameters.update({"num_eps": int(e.value)}))
                    ui.number(label='Number of Steps', value=parameters['nsteps'], 
                             on_change=lambda e: parameters.update({"nsteps": int(e.value)}))
                    ui.number(label='Number of Samples', value=parameters['n_samples'], 
                             on_change=lambda e: parameters.update({"n_samples": int(e.value)}))
                
                # Corrected select calls
                ui.select(options=['Single_Step'], label='Mode', value=parameters['mode'], 
                         on_change=lambda e: parameters.update({"mode": e.value}))
                
                ui.select(options=['circular', 'reflect', 'replicate', 'zeros'], label='Padding Mode', value=parameters['pad_mode'], 
                         on_change=lambda e: parameters.update({"pad_mode": e.value}))
                
                ui.checkbox('Plot During Training', value=parameters['if_plot'], 
                           on_change=lambda e: parameters.update({"if_plot": e.value}))
        
        with ui.tab_panel(grain_tab):
            with ui.card().classes('w-full'):
                ui.label('Grain Parameters').classes('text-xl font-bold')
                
                # Corrected select call
                ui.select(options=['grain', 'circle', 'hex', 'square'], label='Grain Shape', value=parameters['grain_shape'], 
                         on_change=lambda e: parameters.update({"grain_shape": e.value}))
                
                ui.number(label='Grain Size', value=parameters['grain_size'], 
                         on_change=lambda e: parameters.update({"grain_size": int(e.value)}))
                
                ui.checkbox('Voroni Loaded', value=parameters['voroni_loaded'], 
                           on_change=lambda e: parameters.update({"voroni_loaded": e.value}))
                
                ui.input(label='Initial Condition Path', value=parameters['ic'], 
                         on_change=lambda e: parameters.update({"ic": e.value})).classes('w-full')
                
                ui.input(label='Euler Angles Path', value=parameters['ea'], 
                         on_change=lambda e: parameters.update({"ea": e.value})).classes('w-full')
                
                ui.input(label='Misorientation Angles Path', value=parameters['ma'], 
                         on_change=lambda e: parameters.update({"ma": e.value})).classes('w-full')
                
                ui.input(label='IC Shape', value=parameters['ic_shape'], 
                         on_change=lambda e: parameters.update({"ic_shape": e.value})).classes('w-full')
                
                with ui.row():
                    ui.number(label='Dimension', value=parameters['dimension'], 
                             on_change=lambda e: parameters.update({"dimension": int(e.value)}))
                    ui.number(label='Number of Grains', value=parameters['ngrain'], 
                             on_change=lambda e: parameters.update({"ngrain": int(e.value)}))
                
                ui.input(label='PRIMME File (leave empty to run model)', value=parameters['primme'] or "", 
                         on_change=lambda e: parameters.update({"primme": e.value if e.value else None})).classes('w-full')
        
        with ui.tab_panel(run_tab):
            with ui.card().classes('w-full'):
                ui.label('Run Simulation').classes('text-xl font-bold')
                
                console_output = ui.log().classes('w-full h-96 bg-gray-900 text-green-400 font-mono p-4 overflow-auto')
                
                def on_run_click():
                    console_output.clear()
                    console_output.push("Starting PRIMME simulation...")
                    
                    # Run in a separate thread to keep UI responsive
                    thread = threading.Thread(
                        target=run_primme_simulation,
                        args=(parameters, console_output)
                    )
                    thread.daemon = True
                    thread.start()
                
                ui.button('Run Simulation', on_click=on_run_click).classes('bg-blue-500')
        
        with ui.tab_panel(results_tab):
            with ui.card().classes('w-full'):
                ui.label('Results Viewer').classes('text-xl font-bold')
                
                ui.label('This tab will display plots and videos generated by the simulation').classes('italic')
                
                # Placeholder for future implementation
                ui.label('Results viewer functionality will be implemented here')
                
                # This section would scan for output files and display them
                ui.button('Refresh Results', on_click=lambda: ui.notify('Results refresh not implemented yet')).classes('bg-green-500')

# Create the app and then run it
create_app()
ui.run()
