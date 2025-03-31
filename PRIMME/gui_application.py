#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 01:12:31 2026

@author: gabriel.castejon
"""

from nicegui import ui
import subprocess
import sys
import os
from pathlib import Path
import threading
import time
import matplotlib.pyplot as plt
import glob

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))   
os.chdir(__location__) # ensure that the working directory is where this scipt is located.

fp = './data/'
if not os.path.exists(fp): os.makedirs(fp)

fp = './plots/'
if not os.path.exists(fp): os.makedirs(fp)

# Function to run the script with the selected parameters
def run_primme_simulation(parameters, console_output, stop_event):
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

    # Create a separate thread to monitor the stop_event
    def monitor_stop_event():
        while process.poll() is None:  # While process is running
            if stop_event.is_set():
                # Try a more graceful shutdown approach
                try:
                    import signal
                    # Send SIGINT instead of SIGTERM
                    process.send_signal(signal.SIGINT)
                    # Give it some time to clean up
                    time.sleep(0.5)
                    # Only force terminate if it's still running
                    if process.poll() is None:
                        process.terminate()
                except Exception as e:
                    console_output.push(f"Error during termination: {str(e)}")
                console_output.push("Process terminated by user.")
                break
            time.sleep(0.1)  # Check every 100ms
    
    monitor_thread = threading.Thread(target=monitor_stop_event)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Read and display output in real-time
    try:
        for line in iter(process.stdout.readline, ''):
            if stop_event.is_set():
                break
            console_output.push(line)
    except Exception as e:
        console_output.push(f"Error reading output: {str(e)}")
    finally:
        # Ensure resources are properly closed
        try:
            process.stdout.close()
        except:
            pass
    
    return_code = process.wait()
    
    if return_code == 0 and not stop_event.is_set():
        console_output.push("Process completed successfully!")
    elif return_code != 0 and not stop_event.is_set():
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
        "dimension": 2,
        "ngrain": 2**10,
        "primme": None,
        "pad_mode": "circular",
        "if_output_plot": False
    }
    
    data_files = []
    for ext in ['*.h5']:
        data_files.extend(glob.glob(f"./data/{ext}"))
    spparks_trainsets = [f for f in data_files if 'spparks' in f]
    models_trained = [f for f in data_files if 'model' in f]
    primme_simulations = [f for f in data_files if 'primme' in f]
    
    # Create UI
    ui.switch('Dark Mode', value=False, on_change=lambda e: ui.dark_mode().enable() if e.value else ui.dark_mode().disable())
    
    with ui.tabs().classes('w-full') as tabs:
        model_tab = ui.tab('Training Parameters')
        grain_tab = ui.tab('Testing Parameters')
        run_tab = ui.tab('Run Simulation')
        results_tab = ui.tab('Results')
    
    with ui.tab_panels(tabs, value=model_tab).classes('w-full'):
        with ui.tab_panel(model_tab):
            with ui.card().classes('w-full'):
                ui.label('Training Parameters').classes('text-xl font-bold')
                
                ui.select(options=spparks_trainsets, label='Training Set', value=parameters['trainset'], 
                         on_change=lambda e: parameters.update({"trainset": e.value})).classes('w-full')
                
                model_name = ui.select(options=['Train New Model'] + models_trained, label='Model Name', value=parameters['modelname'] or 'Train New Model', 
                         on_change=lambda e: parameters.update({"modelname": e.value if e.value != 'Train New Model' else None})).classes('w-full')
                

                with ui.column().classes('w-full gap-4') as model_params:
                    with ui.row().classes('gap-4 items-center justify-between'):
                        # ui.select(options=[2, 3], label='Training Dimensions', value=parameters['dims'], 
                        #         on_change=lambda e: parameters.update({"dims": int(e.value)})).classes('w-auto min-w-[150px]')
                        ui.select(options=[7,9,11,13,15,17,19,21], label='Observation Dimension', value=parameters['obs_dim'], 
                                on_change=lambda e: parameters.update({"obs_dim": int(e.value)})).classes('w-auto min-w-[150px]')
                        ui.select(options=[7,9,11,13,15,17,19,21], label='Action Dimension', value=parameters['act_dim'], 
                                on_change=lambda e: parameters.update({"act_dim": int(e.value)})).classes('w-auto min-w-[150px]')

                
                # Leaving out customization for these:
                # with ui.row():
                #     ui.number(label='Learning Rate', value=parameters['lr'], format='%.5f', 
                #              on_change=lambda e: parameters.update({"lr": float(e.value)}))
                #     ui.number(label='Regularization', value=parameters['reg'], 
                #              on_change=lambda e: parameters.update({"reg": float(e.value)}))
                
                    with ui.row():
                        def update_num_eps(e):
                            parameters.update({"num_eps": int(e.value)})
                            num_eps_label.set_text(f"Number of Training Epochs: {e.value}")
                        
                        num_eps_label = ui.label(f"Number of Training Epochs: {parameters['num_eps']}").classes('font-bold')
                        ui.slider(min=5, max=2000, step=5, value=parameters['num_eps'], 
                                on_change=update_num_eps).classes('w-full -mt-5')
                        
                    ui.select(options=['circular', 'reflect'], label='Padding Mode', value=parameters['pad_mode'], 
                         on_change=lambda e: parameters.update({"pad_mode": e.value}))
                
                    ui.checkbox('Output Plots During Training', value=parameters['if_plot'], 
                            on_change=lambda e: parameters.update({"if_plot": e.value}))
                    
                model_params.bind_visibility_from(model_name, 'value', lambda v: v == 'Train New Model')
                # The two below options are generally not changed so leaving out.
                    # ui.number(label='Number of Samples', value=parameters['n_samples'], 
                    #          on_change=lambda e: parameters.update({"n_samples": int(e.value)}))
                
                # ui.select(options=['Single_Step'], label='Mode', value=parameters['mode'], 
                #          on_change=lambda e: parameters.update({"mode": e.value}))
                
                
                # Can add later, but not intuitive right now.
                # ui.checkbox('Output Plots After Simulation', value=parameters['if_output_plot'],
                #             on_change=lambda e: parameters.update({"if_output_plot": e.value}))
        
        with ui.tab_panel(grain_tab):
            with ui.card().classes('w-full'):
                ui.label('Testing Parameters').classes('text-xl font-bold')

                primme_selected = ui.select(options=["Run New Model"] + primme_simulations, label='PRIMME Simulation', value=parameters['primme'] or 'Run New Model',
                            on_change=lambda e: parameters.update({"primme": e.value if e.value != 'Run New Model' else None})).classes('w-full')
                
                with ui.row().classes('w-full') as grain_params:
                    voroni_checkbox = ui.checkbox('Voroni Loaded', value=parameters['voroni_loaded'], 
                    on_change=lambda e: parameters.update({"voroni_loaded": e.value}))
                    
                    with ui.column().classes('w-full') as non_loaded_inputs:
                        grain_size_select = ui.select(options=[257, 512, 1024, 2048, 2400], label='Grain Size', value=parameters['grain_size'], 
                                on_change=lambda e: parameters.update({"grain_size": int(e.value)})).classes('!w-full')
        
                        ui.select(options=['grain', 'circular', 'hex', 'square'], label='Grain Shape', value=parameters['grain_shape'], 
                                on_change=lambda e: (parameters.update({"grain_shape": e.value}), update_grain(e))).classes('!w-full')
            
                    # Create the input fields that should be conditionally visible
                    # Use a container to group them for easier visibility control
                    with ui.column() as loaded_inputs:
                        ui.input(label='Initial Condition Path', value=parameters['ic'], 
                                on_change=lambda e: parameters.update({"ic": e.value})).classes('w-full')
                        
                        ui.input(label='Euler Angles Path', value=parameters['ea'], 
                                on_change=lambda e: parameters.update({"ea": e.value})).classes('w-full')
                        
                        ui.input(label='Miso Angles Path', value=parameters['ma'], 
                                on_change=lambda e: parameters.update({"ma": e.value})).classes('w-full')
                    
                            # Bind the visibility of the path inputs to the checkbox value
                    loaded_inputs.bind_visibility_from(voroni_checkbox, 'value')
                    non_loaded_inputs.bind_visibility_from(voroni_checkbox, 'value', backward=lambda x: not x)

                    with ui.row():
                        def update_ngrain(e):
                            exponent = int(e.value)
                            ngrain = 2 ** exponent
                            parameters.update({"ngrain": ngrain})
                            ngrain_label.set_text(f"Number of Grains: 2^{exponent} or {ngrain}")
                        
                        ngrain_label = ui.label(f"Number of Grains: 2^10 or {parameters['ngrain']}").classes('font-bold')
                        ngrain_slider = ui.slider(min=6, max=18, value=parameters['ngrain'].bit_length() - 1, 
                                            on_change=update_ngrain).classes('w-full -mt-5')
                        
                        def update_num_steps(e):
                            parameters.update({"nsteps": int(e.value)})
                            num_steps_label.set_text(f"Number of Steps: {e.value}")
                        
                        num_steps_label = ui.label(f"Number of Steps: {parameters['nsteps']}").classes('font-bold')
                        ui.slider(min=10, max=1000, step=5, value=parameters['nsteps'], 
                                on_change=update_num_steps).classes('w-full -mt-5')

                        # Update the grain shape change handler to also update the ngrain slider
                        def update_grain(e):
                            if e.value == 'grain':
                                # Enable the slider and reset to default value
                                ngrain_slider.set_enabled(True)
                                ngrain_slider.value = 10
                                parameters.update({"ngrain": 2**10})
                                ngrain_label.set_text("Number of Grains: 2^10 or 1024")
                                ngrain_slider.update()

                                grain_size_select.enable()
                                grain_size_select.set_value(512)
                                grain_size_select.update()
                            else:
                                if e.value == 'hex':
                                    grain_size_select.disable()
                                    grain_size_select.update()
                                if e.value == 'circular':
                                    grain_size_select.set_value(257)
                                    grain_size_select.disable()
                                    grain_size_select.update()
                                # Disable the slider and set ngrain to 10
                                # ngrain_slider.set_enabled(False)
                                # parameters.update({"ngrain": 10})
                                # ngrain_label.set_text("Number of Grains: 10")
                                # ngrain_slider.update()  # Update the slider to reflect the change   
                grain_params.bind_visibility_from(primme_selected, 'value', lambda v: v == 'Run New Model')
                # IC Shape does not seem intuitive so leaving out.
                # ui.input(label='IC Shape', value=parameters['ic_shape'], 
                #          on_change=lambda e: parameters.update({"ic_shape": e.value})).classes('w-full')
                
                     
        
        with ui.tab_panel(run_tab):
            with ui.card().classes('w-full'):
                ui.label('Run Simulation').classes('text-xl font-bold')
                
                console_output = ui.log().classes('w-full h-96 bg-gray-900 text-green-400 font-mono p-4 overflow-auto')
                
                stop_event = threading.Event()

                def on_run_click():
                    console_output.clear()
                    console_output.push("Starting PRIMME simulation...")
                    stop_event.clear()
                    
                    # Run in a separate thread to keep UI responsive
                    thread = threading.Thread(
                        target=run_primme_simulation,
                        args=(parameters, console_output, stop_event)
                    )
                    thread.daemon = True
                    thread.start()

                def on_run_stop():
                    console_output.push("Stopping PRIMME simulation...")
                    stop_event.set()
                
                with ui.row().classes('w-full gap-4 items-center justify-between'):
                    run_button = ui.button('Run Simulation').classes('!bg-blue-500')
                    stop_button = ui.button('Stop Simulation').classes('!bg-red-500')

                def update_buttons(button=0):
                    run_button.set_enabled(not button)
                    stop_button.set_enabled(button)

                run_button.on('click', lambda: (on_run_click(), update_buttons(1)))
                stop_button.on('click', lambda: (on_run_stop(), update_buttons(0)))

                update_buttons()
        
        with ui.tab_panel(results_tab):
            with ui.card().classes('w-full'):
                # Header row with title and refresh button side by side
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('Results Viewer').classes('text-xl font-bold')
                    ui.button('Refresh Results', on_click=lambda: (refresh_results(), refresh_videos())).classes('bg-green-500')
                
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
                                
                video_container = ui.column().classes('w-full gap-4 items-center')
                
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
                                
                # Initial load of results
                refresh_results()
                refresh_videos()


# Create the app and then run it
create_app()
ui.run()
