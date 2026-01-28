import subprocess
import threading
import sys
import os
import signal
import json
from collections import deque

class ExportManager:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.logs = deque(maxlen=2000)
        self.status = "idle" # idle, running, completed, error, stopping
        self.config = {}
        self.output_file = None
        self.monitor_thread = None

    def start_export(self, config):
        with self.lock:
            if self.status == "running":
                raise Exception("Export is already running")
            
            self.status = "running"
            self.config = config
            self.logs.clear()
            self.output_file = None
            
            # Determine python executable
            python_exe = sys.executable
            worker_script = os.path.join(os.path.dirname(__file__), "export_worker.py")
            
            # Serialize config
            config_json = json.dumps(config)
            
            # Start subprocess
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

    def stop_export(self):
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
                    self.logs.append(clean_line)
                    
                    if clean_line.startswith("EXPORT_SUCCESS: "):
                        self.output_file = clean_line.replace("EXPORT_SUCCESS: ", "").strip()

            self.process.stdout.close()
            return_code = self.process.wait()
            
            with self.lock:
                if return_code == 0:
                    self.status = "completed"
                    self.logs.append("Export Process Finished Successfully.")
                else:
                    if self.status == "stopping":
                        self.status = "canceled"
                        self.logs.append("Export Process Canceled by User.")
                    else:
                        self.status = "error"
                        self.logs.append(f"Export Process Failed with exit code {return_code}")
                
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
                "output_file": self.output_file
            }

export_manager = ExportManager()
