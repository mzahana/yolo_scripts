import subprocess
import threading
import sys
import os
import signal
import json
import time
from collections import deque
from pathlib import Path

class TrainingManager:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.logs = deque(maxlen=2000) # Keep last 2000 lines
        self.status = "idle" # idle, running, completed, error, stopping
        self.config = {}
        self.current_epoch = 0
        self.total_epochs = 0
        self.result_dir = None
        self.result_summary = None
        self._stop_event = threading.Event()
        self.monitor_thread = None

    def start_training(self, config):
        with self.lock:
            if self.status == "running":
                raise Exception("Training is already running")
            
            self.status = "running"
            self.config = config
            self.logs.clear()
            self._stop_event.clear()
            
            # Determine python executable
            python_exe = sys.executable
            worker_script = os.path.join(os.path.dirname(__file__), "train_worker.py")
            
            # Serialize config
            config_json = json.dumps(config)
            
            # Start subprocess
            # usage of setsid to allow killing the whole process group if needed
            self.process = subprocess.Popen(
                [python_exe, "-u", worker_script, "--config", config_json],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1, # Line buffered
                preexec_fn=os.setsid 
            )
            
            self.monitor_thread = threading.Thread(target=self._monitor_process)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            return {"status": "started", "pid": self.process.pid}

    def stop_training(self):
        with self.lock:
            if self.status != "running" or not self.process:
                return {"status": "not_running"}
            
            self.status = "stopping"
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except Exception as e:
                self.logs.append(f"Error stopping process: {e}")
            
            return {"status": "stopping"}

    def _monitor_process(self):
        try:
            # Read stdout line by line
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    clean_line = line.strip()
                    # ANSI escape code stripping could be added here if needed, 
                    # but pure text is often fine or handled by frontend terminal
                    self.logs.append(clean_line)
                    
                    if clean_line.startswith("RESULT_DIR: "):
                        self.result_dir = clean_line.replace("RESULT_DIR: ", "").strip()
                        
                    if clean_line.startswith("RESULT_SUMMARY: "):
                        try:
                            self.result_summary = json.loads(clean_line.replace("RESULT_SUMMARY: ", ""))
                        except: pass

                    # Basic parsing for progress (epochs)
                    # format usually:  1/100 ...
                    if "/" in clean_line and "Epoch" in clean_line:
                        # Try to parse epoch
                        pass 

            self.process.stdout.close()
            return_code = self.process.wait()
            
            with self.lock:
                if return_code == 0:
                    self.status = "completed"
                    self.logs.append("Training Process Finished Successfully.")
                else:
                    if self.status == "stopping":
                        self.status = "canceled"
                        self.logs.append("Training Process Canceled by User.")
                    else:
                        self.status = "error"
                        self.logs.append(f"Training Process Failed with exit code {return_code}")
                
                self.process = None

        except Exception as e:
            with self.lock:
                self.status = "error"
                self.logs.append(f"Internal Monitor Error: {e}")

    def get_state(self):
        with self.lock:
            return {
                "status": self.status,
                "config": self.config,
                "logs": list(self.logs),
                "is_active": self.status == "running",
                "result_dir": self.result_dir,
                "result_summary": self.result_summary
            }

training_manager = TrainingManager()
