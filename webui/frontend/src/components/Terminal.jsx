import React, { useState } from 'react';
import TerminalInstance from './TerminalInstance';

const WebTerminal = () => {
    const [terminals, setTerminals] = useState([{ id: 1, name: 'Terminal 1' }]);
    const [activeTerminalId, setActiveTerminalId] = useState(1);
    const [nextId, setNextId] = useState(2);

    const addTerminal = () => {
        const newId = nextId;
        setTerminals([...terminals, { id: newId, name: `Terminal ${newId}` }]);
        setActiveTerminalId(newId);
        setNextId(nextId + 1);
    };

    const closeTerminal = (e, id) => {
        e.stopPropagation(); // Prevent tab switching when closing
        const newTerminals = terminals.filter(t => t.id !== id);

        if (newTerminals.length === 0) {
            // Keep at least one terminal or handle empty state.
            // Let's reset to a single new terminal.
            setTerminals([{ id: nextId, name: `Terminal ${nextId}` }]);
            setActiveTerminalId(nextId);
            setNextId(nextId + 1);
        } else {
            setTerminals(newTerminals);
            // If we closed the active one, switch to the last one
            if (activeTerminalId === id) {
                setActiveTerminalId(newTerminals[newTerminals.length - 1].id);
            }
        }
    };

    return (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {/* Tabs Header */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px', borderBottom: '1px solid #333', paddingBottom: '5px', alignItems: 'center' }}>
                {terminals.map(term => (
                    <div
                        key={term.id}
                        onClick={() => setActiveTerminalId(term.id)}
                        style={{
                            padding: '6px 15px',
                            background: activeTerminalId === term.id ? '#374151' : 'rgba(255,255,255,0.05)',
                            color: activeTerminalId === term.id ? 'white' : '#aaa',
                            borderRadius: '6px 6px 0 0',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '10px',
                            border: '1px solid transparent',
                            borderColor: activeTerminalId === term.id ? '#4b5563' : 'transparent',
                            borderBottom: 'none',
                            userSelect: 'none'
                        }}
                    >
                        <span>{term.name}</span>
                        <span
                            onClick={(e) => closeTerminal(e, term.id)}
                            style={{
                                opacity: 0.5,
                                cursor: 'pointer',
                                fontSize: '0.8rem',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                width: '16px', height: '16px',
                                borderRadius: '50%'
                            }}
                            onMouseEnter={e => e.currentTarget.style.backgroundColor = '#555'}
                            onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}
                        >
                            ✕
                        </span>
                    </div>
                ))}

                <button
                    onClick={addTerminal}
                    style={{
                        background: 'transparent',
                        border: '1px solid #444',
                        color: '#aaa',
                        width: '30px',
                        height: '30px',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '1.2rem',
                        marginLeft: '5px'
                    }}
                    title="New Terminal"
                >
                    +
                </button>
            </div>

            {/* Terminal Instances Area */}
            <div style={{ flex: 1, minHeight: 0, position: 'relative' }}>
                {terminals.map(term => (
                    <div
                        key={term.id}
                        style={{
                            height: '100%',
                            display: activeTerminalId === term.id ? 'block' : 'none'
                        }}
                    >
                        <TerminalInstance visible={activeTerminalId === term.id} />
                    </div>
                ))}
            </div>
        </div>
    );
};

export default WebTerminal;
