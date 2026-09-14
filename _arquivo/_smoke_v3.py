import sys, os, time, runpy
sys.path.insert(0, os.getcwd())
mod = runpy.run_path("music_maker_ui_v3.py", run_name="not_main")
demo = mod["demo"]
demo.launch(server_name="127.0.0.1", server_port=7854, prevent_thread_lock=True)
time.sleep(6)
print("LAUNCHED_OK")
