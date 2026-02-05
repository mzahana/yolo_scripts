import os
import pty
import select
import struct
import fcntl
import termios
import signal
import threading
import time
from typing import Optional, Tuple, Dict

class TerminalManager:
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        
    def start_session(self, session_id: str) -> Tuple[int, int]:
        """
        Starts a new pseudo-terminal session.
        Returns (master_fd, slave_fd)
        """
        # Create a new pty
        master_fd, slave_fd = pty.openpty()
        
        # Fork the process
        pid = os.fork()
        
        if pid == 0:
            # Child process
            os.setsid()
            os.dup2(slave_fd, 0)
            os.dup2(slave_fd, 1)
            os.dup2(slave_fd, 2)
            
            # Close file descriptors
            if master_fd > 2:
                os.close(master_fd)
            if slave_fd > 2:
                os.close(slave_fd)
                
            # Set terminal environment
            env = os.environ.copy()
            env["TERM"] = "xterm-256color"
            env["COLUMNS"] = "80"
            env["LINES"] = "24"
            
            # Exec shell
            shell = os.environ.get('SHELL', '/bin/bash')
            os.execvpe(shell, [shell], env)
        else:
            # Parent process
            self.active_sessions[session_id] = {
                "master_fd": master_fd,
                "pid": pid
            }
            return master_fd, pid

    def resize_terminal(self, session_id: str, cols: int, rows: int):
        """
        Resizes the terminal window size.
        """
        if session_id in self.active_sessions:
            fd = self.active_sessions[session_id]["master_fd"]
            # Set terminal size
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            try:
                fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)
            except OSError:
                pass

    def write_input(self, session_id: str, data: str):
        """
        Writes input data to the terminal.
        """
        if session_id in self.active_sessions:
            fd = self.active_sessions[session_id]["master_fd"]
            try:
                os.write(fd, data.encode())
            except OSError:
                pass

    def read_output(self, session_id: str, max_bytes: int = 1024, timeout: float = 0.1) -> Optional[bytes]:
        """
        Reads output from the terminal. 
        Uses select to wait for data with a timeout, preventing indefinite blocking.
        """
        if session_id in self.active_sessions:
            fd = self.active_sessions[session_id]["master_fd"]
            try:
                # Check if data is available to read
                r, _, _ = select.select([fd], [], [], timeout)
                if fd in r:
                    data = os.read(fd, max_bytes)
                    if not data: # EOF
                        return None
                    return data
                return None # Timeout
            except OSError:
                return None
        return None

    def close_session(self, session_id: str):
        """
        Terminates the session and cleans up.
        """
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            fd = session["master_fd"]
            pid = session["pid"]
            
            # Close file descriptor
            try:
                os.close(fd)
            except OSError:
                pass
                
            # Kill process
            try:
                os.kill(pid, signal.SIGTERM)
                # Wait for process to terminate to avoid zombies
                os.waitpid(pid, 0)
            except OSError:
                pass
                
            del self.active_sessions[session_id]

# Singleton instance
terminal_manager = TerminalManager()
