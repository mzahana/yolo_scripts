import React, { useEffect, useRef, useState } from 'react';
import { Terminal } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import 'xterm/css/xterm.css';

const WebTerminal = () => {
    const terminalContainerRef = useRef(null);
    const wsRef = useRef(null);
    const xtermRef = useRef(null);
    const fitAddonRef = useRef(null);
    const [status, setStatus] = useState('disconnected');
    const [logs, setLogs] = useState([]);

    const addLog = (msg) => {
        const time = new Date().toISOString().split('T')[1].split('.')[0];
        setLogs(prev => [`[${time}] ${msg}`, ...prev].slice(0, 50));
        console.log(`[Terminal] ${msg}`);
    };

    const connect = () => {
        if (wsRef.current) {
            addLog("Closing existing WS before new connection");
            wsRef.current.close();
        }

        setStatus('connecting');
        addLog("Starting connection process...");

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Connect directly to backend port 8000 to avoid proxy issues
        const hostname = window.location.hostname;
        const wsUrl = `${protocol}//${hostname}:8000/api/ws/terminal`;

        addLog(`Connecting to: ${wsUrl}`);

        try {
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                if (wsRef.current !== ws) {
                    addLog("Ignoring stale WS open event");
                    return;
                }
                addLog("WebSocket Open");
                setStatus('connected');

                if (xtermRef.current) {
                    xtermRef.current.write('\r\n\x1b[32mTarget Connected\x1b[0m\r\n\r\n');
                }

                // Initial resize
                if (fitAddonRef.current) {
                    try {
                        const dims = fitAddonRef.current.proposeDimensions();
                        if (dims) {
                            addLog(`Sending resize: ${dims.cols}x${dims.rows}`);
                            ws.send(JSON.stringify({ type: 'resize', cols: dims.cols, rows: dims.rows }));
                        }
                    } catch (e) {
                        addLog(`Error proposing dimensions: ${e.message}`);
                    }
                }
            };

            ws.onmessage = (event) => {
                if (wsRef.current !== ws) return;
                if (xtermRef.current) {
                    xtermRef.current.write(event.data);
                }
            };

            ws.onclose = (event) => {
                if (wsRef.current !== ws) return;
                addLog(`WebSocket Closed: Code=${event.code}, Reason=${event.reason || 'None'}`);
                setStatus('disconnected');
            };

            ws.onerror = (error) => {
                if (wsRef.current !== ws) return;
                addLog("WebSocket Error occurred");
                setStatus('error');
            };

        } catch (e) {
            addLog(`Exception creating WebSocket: ${e.message}`);
            setStatus('error');
        }
    };

    useEffect(() => {
        addLog("Component MOUNTED");

        if (!terminalContainerRef.current) {
            addLog("Container ref is null!");
            return;
        }

        // Initialize xterm
        const term = new Terminal({
            cursorBlink: true,
            theme: {
                background: '#1e1e1e',
                foreground: '#ffffff',
            },
            fontFamily: 'Menlo, Monaco, "Courier New", monospace',
            fontSize: 14,
        });

        const fitAddon = new FitAddon();
        term.loadAddon(fitAddon);

        try {
            term.open(terminalContainerRef.current);
            fitAddon.fit();
            xtermRef.current = term;
            fitAddonRef.current = fitAddon;
            addLog("xterm initialized");
        } catch (e) {
            addLog(`Error initializing xterm: ${e.message}`);
        }

        // Connect after a short delay to allow layout to settle
        setTimeout(connect, 100);

        // Input handler
        const handleData = (data) => {
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ type: 'input', data: data }));
            }
        };
        const disposable = term.onData(handleData);

        // Resize handler
        const handleResize = () => {
            if (fitAddonRef.current) {
                try {
                    fitAddonRef.current.fit();
                    const dims = fitAddonRef.current.proposeDimensions();
                    if (dims && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                        wsRef.current.send(JSON.stringify({ type: 'resize', cols: dims.cols, rows: dims.rows }));
                    }
                } catch (e) {
                    // console.error(e);
                }
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            addLog("Component UNMOUNTING");
            window.removeEventListener('resize', handleResize);
            disposable.dispose();

            if (wsRef.current) {
                wsRef.current.close(1000, "Unmounting");
                wsRef.current = null;
            }
            if (xtermRef.current) {
                xtermRef.current.dispose();
                xtermRef.current = null;
            }
        };
    }, []);

    const handleReconnect = () => {
        if (xtermRef.current) {
            xtermRef.current.reset();
        }
        connect();
    };

    return (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: '#ccc' }}>
                <div style={{ fontWeight: 'bold' }}>Terminal</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                        <div style={{
                            width: '10px', height: '10px', borderRadius: '50%',
                            backgroundColor: status === 'connected' ? '#10b981' : (status === 'connecting' ? '#f59e0b' : '#ef4444')
                        }}></div>
                        <span style={{ fontSize: '0.9rem' }}>{status}</span>
                    </div>
                    {status !== 'connected' && (
                        <button
                            onClick={handleReconnect}
                            style={{
                                padding: '4px 12px',
                                background: '#374151',
                                border: 'none',
                                borderRadius: '4px',
                                color: 'white',
                                cursor: 'pointer'
                            }}
                        >
                            Reconnect
                        </button>
                    )}
                </div>
            </div>

            {/* Terminal View */}
            <div
                ref={terminalContainerRef}
                className="terminal-container"
                style={{
                    flex: 1,
                    minHeight: '400px',
                    background: '#1e1e1e',
                    borderRadius: '8px',
                    padding: '10px',
                    overflow: 'hidden'
                }}
            />

            {/* Debug Logs */}
            <div style={{
                height: '150px',
                overflowY: 'auto',
                background: '#111',
                color: '#22c55e',
                fontFamily: 'monospace',
                fontSize: '0.8rem',
                padding: '10px',
                borderRadius: '8px',
                border: '1px solid #333'
            }}>
                <div style={{ color: '#888', marginBottom: '5px', fontWeight: 'bold' }}>Debug Logs (Latest First):</div>
                {logs.map((log, i) => (
                    <div key={i}>{log}</div>
                ))}
            </div>
        </div>
    );
};

export default WebTerminal;
