import React, { useEffect, useRef, useState } from 'react';
import { Terminal } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import 'xterm/css/xterm.css';

const TerminalInstance = ({ visible }) => {
    const terminalContainerRef = useRef(null);
    const wsRef = useRef(null);
    const xtermRef = useRef(null);
    const fitAddonRef = useRef(null);
    const [status, setStatus] = useState('disconnected');
    const [logs, setLogs] = useState([]);

    const addLog = (msg) => {
        const time = new Date().toISOString().split('T')[1].split('.')[0];
        setLogs(prev => [`[${time}] ${msg}`, ...prev].slice(0, 50));
        // console.log(`[Terminal] ${msg}`);
    };

    const connect = () => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        setStatus('connecting');
        addLog("Connecting...");

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Fix: Use window.location.host (includes port) to go through the Vite proxy
        // This ensures it works when accessed via SSH tunnel on a different port (e.g. 3002)
        const wsUrl = `${protocol}//${window.location.host}/api/ws/terminal`;

        try {
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                if (wsRef.current !== ws) return;
                addLog("Connected");
                setStatus('connected');

                if (xtermRef.current) {
                    xtermRef.current.write('\r\n\x1b[32mTarget Connected\x1b[0m\r\n\r\n');
                }

                // Initial resize
                if (fitAddonRef.current) {
                    setTimeout(() => {
                        try {
                            fitAddonRef.current.fit();
                            const dims = fitAddonRef.current.proposeDimensions();
                            if (dims) {
                                ws.send(JSON.stringify({ type: 'resize', cols: dims.cols, rows: dims.rows }));
                            }
                        } catch (e) { console.error(e); }
                    }, 100);
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
                addLog(`Closed: ${event.code}`);
                setStatus('disconnected');
            };

            ws.onerror = (error) => {
                if (wsRef.current !== ws) return;
                addLog("Error occurred");
                setStatus('error');
            };

        } catch (e) {
            addLog(`Exception: ${e.message}`);
            setStatus('error');
        }
    };

    useEffect(() => {
        if (!terminalContainerRef.current) return;

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
        } catch (e) {
            console.error(e);
        }

        // Connect after a short delay
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
                } catch (e) { }
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
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

    // Re-fit when becoming visible
    useEffect(() => {
        if (visible && fitAddonRef.current) {
            try {
                fitAddonRef.current.fit();
                // We might need to send a resize event here too if dimensions changed while hidden
                const dims = fitAddonRef.current.proposeDimensions();
                if (dims && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                    wsRef.current.send(JSON.stringify({ type: 'resize', cols: dims.cols, rows: dims.rows }));
                }
            } catch (e) { }
        }
    }, [visible]);

    const handleReconnect = () => {
        if (xtermRef.current) {
            xtermRef.current.reset();
        }
        connect();
    };

    return (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '5px' }}>
            <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', height: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: '#aaa' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <div style={{
                            width: '8px', height: '8px', borderRadius: '50%',
                            backgroundColor: status === 'connected' ? '#10b981' : (status === 'connecting' ? '#f59e0b' : '#ef4444')
                        }}></div>
                        <span>{status}</span>
                    </div>
                    {status !== 'connected' && (
                        <button
                            onClick={handleReconnect}
                            style={{
                                padding: '2px 8px',
                                background: '#374151',
                                border: 'none',
                                borderRadius: '4px',
                                color: 'white',
                                cursor: 'pointer',
                                fontSize: '0.75rem'
                            }}
                        >
                            Reconnect
                        </button>
                    )}
                </div>
            </div>

            <div
                ref={terminalContainerRef}
                className="terminal-container"
                style={{
                    flex: 1,
                    minHeight: 0,
                    background: '#1e1e1e',
                    borderRadius: '4px',
                    padding: '5px',
                    overflow: 'hidden'
                }}
            />
        </div>
    );
};

export default TerminalInstance;
