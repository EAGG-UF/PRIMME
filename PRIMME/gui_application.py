from nicegui import ui
import subprocess
import sys
import os
from pathlib import Path
import threading
import time
import matplotlib.pyplot as plt
import glob

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

# Display each plot
def format_plot_title(filename):
    stem = Path(filename).stem
    
    # Remove everything before and including the closing parenthesis if it exists
    if ')' in stem:
        stem = stem.split(')')[-1]
    
    # Replace underscores with spaces and capitalize
    return stem.replace('_', ' ').title()
        
# Main application
def create_app():
    # Initialize parameters with defaults
    parameters = {
        "trainset": "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5",
        "modelname": "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5",
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
        "pad_mode": "circular",
        "if_output_plot": False
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
                    ui.select(options=[2, 3], label='Training Dimensions', value=parameters['dims'], 
                             on_change=lambda e: parameters.update({"dims": int(e.value)})).classes('w-full')
                    ui.number(label='Observation Dimension', value=parameters['obs_dim'], 
                             on_change=lambda e: parameters.update({"obs_dim": int(e.value)}))
                    ui.number(label='Action Dimension', value=parameters['act_dim'], 
                             on_change=lambda e: parameters.update({"act_dim": int(e.value)}))
                
                # Leaving out customization for these:
                # with ui.row():
                #     ui.number(label='Learning Rate', value=parameters['lr'], format='%.5f', 
                #              on_change=lambda e: parameters.update({"lr": float(e.value)}))
                #     ui.number(label='Regularization', value=parameters['reg'], 
                #              on_change=lambda e: parameters.update({"reg": float(e.value)}))
                
                with ui.row():
                    ui.number(label='Number of Epochs', value=parameters['num_eps'], 
                             on_change=lambda e: parameters.update({"num_eps": int(e.value)}))
                    ui.number(label='Number of Steps', value=parameters['nsteps'], 
                             on_change=lambda e: parameters.update({"nsteps": int(e.value)}))
                    
                # The two below options are generally not changed so leaving out.
                    # ui.number(label='Number of Samples', value=parameters['n_samples'], 
                    #          on_change=lambda e: parameters.update({"n_samples": int(e.value)}))
                
                # ui.select(options=['Single_Step'], label='Mode', value=parameters['mode'], 
                #          on_change=lambda e: parameters.update({"mode": e.value}))
                
                ui.select(options=['circular', 'reflect'], label='Padding Mode', value=parameters['pad_mode'], 
                         on_change=lambda e: parameters.update({"pad_mode": e.value}))
                
                ui.checkbox('Output Plots During Training', value=parameters['if_plot'], 
                           on_change=lambda e: parameters.update({"if_plot": e.value}))
                # Can add later, but not intuitive right now.
                # ui.checkbox('Output Plots After Simulation', value=parameters['if_output_plot'],
                #             on_change=lambda e: parameters.update({"if_output_plot": e.value}))
        
        with ui.tab_panel(grain_tab):
            with ui.card().classes('w-full'):
                ui.label('Grain Parameters').classes('text-xl font-bold')
                
                # Corrected select call
                ui.select(options=['grain', 'circle', 'hex', 'square'], label='Grain Shape', value=parameters['grain_shape'], 
                         on_change=lambda e: parameters.update({"grain_shape": e.value})).classes('w-full')
                
                ui.number(label='Grain Size', value=parameters['grain_size'], 
                         on_change=lambda e: parameters.update({"grain_size": int(e.value)}))
                
                voroni_checkbox = ui.checkbox('Voroni Loaded', value=parameters['voroni_loaded'], 
                   on_change=lambda e: parameters.update({"voroni_loaded": e.value}))
        
                # Create the input fields that should be conditionally visible
                # Use a container to group them for easier visibility control
                with ui.column() as path_inputs:
                    ic_input = ui.input(label='Initial Condition Path', value=parameters['ic'], 
                            on_change=lambda e: parameters.update({"ic": e.value})).classes('w-full')
                    
                    ea_input = ui.input(label='Euler Angles Path', value=parameters['ea'], 
                            on_change=lambda e: parameters.update({"ea": e.value})).classes('w-full')
                    
                    ma_input = ui.input(label='Misorientation Angles Path', value=parameters['ma'], 
                            on_change=lambda e: parameters.update({"ma": e.value})).classes('w-full')
                
                        # Bind the visibility of the path inputs to the checkbox value
                path_inputs.bind_visibility_from(voroni_checkbox, 'value')

                # IC Shape does not seem intuitive so leaving out.
                # ui.input(label='IC Shape', value=parameters['ic_shape'], 
                #          on_change=lambda e: parameters.update({"ic_shape": e.value})).classes('w-full')
                
                with ui.row():
                    ui.select(options=[2, 3], label='Dimension', value=parameters['dimension'], 
                             on_change=lambda e: parameters.update({"dimension": int(e.value)})).classes('w-full')

                    def update_ngrain(e):
                        exponent = int(e.value)
                        ngrain = 2 ** exponent
                        parameters.update({"ngrain": ngrain})
                        ngrain_label.set_text(f"Number of Grains: 2^{exponent} or {ngrain}")

                    ui.slider(min=6, max=18, value=parameters['ngrain'].bit_length() - 1, 
                              on_change=update_ngrain).classes('w-full')
                    ngrain_label = ui.label(f"Number of Grains: 2^14 or {parameters['ngrain']}").classes('ml-4')
                
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
                # Header row with title and refresh button side by side
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('Results Viewer').classes('text-xl font-bold')
                    ui.button('Refresh All Results', on_click=lambda: (refresh_results(), refresh_videos())).classes('bg-green-500')
                
                # Create a container for plots
                plot_container = ui.column().classes('w-full gap-4 items-center')  # Added items-center to center content
                
                def refresh_results():
                    # Clear existing plots
                    plot_container.clear()
                    
                    # Look for plot files (assuming they're saved as PNG or similar)
                    plot_files = []
                    for ext in ['*.png', '*.jpg', '*.jpeg']:
                        plot_files.extend(glob.glob(f"./plots/{ext}"))
                    
                    if not plot_files:
                        with plot_container:
                            ui.label('No plots found').classes('italic text-gray-500')
                        return
                
                    for plot_file in sorted(plot_files):
                        with plot_container:
                            ui.label(format_plot_title(plot_file)).classes('font-bold text-center')
                            # Center the image with a flex container
                            with ui.row().classes('w-full justify-center'):
                                ui.image(plot_file).classes('max-w-2xl')
                
                # Add a section for video display
                with ui.expansion('Video Results', icon='movie').classes('w-full mt-4'):
                    video_container = ui.column().classes('w-full gap-4 items-center')  # Added items-center
                    
                    def refresh_videos():
                        video_container.clear()
                        video_files = glob.glob('./plots/*.mp4')
                        
                        if not video_files:
                            with video_container:
                                ui.label('No videos found').classes('italic text-gray-500')
                            return
                        
                        for video_file in sorted(video_files):
                            with video_container:
                                ui.label(format_plot_title(video_file)).classes('font-bold text-center')
                                # Center the video with a flex container
                                with ui.row().classes('w-full justify-center'):
                                    ui.video(video_file).classes('max-w-2xl')
                    
                    ui.button('Refresh Videos', on_click=refresh_videos).classes('mt-2')
                
                # Initial load of results
                refresh_results()
                refresh_videos()


# Create the app and then run it
create_app()
ui.run()
