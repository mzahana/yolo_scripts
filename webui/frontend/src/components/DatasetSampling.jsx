import React, { useState, useEffect } from 'react';
import ProgressBar from './ProgressBar';

const DatasetSampling = ({
    initialPath,
    projectConfig,
    onBrowse,
    showNotification,
    isTaskRunning,
    taskProgress,
    onStartTask
}) => {
    const [sourcePath, setSourcePath] = useState(initialPath || '');
    const [outputPath, setOutputPath] = useState('');
    const [samplingMode, setSamplingMode] = useState('global'); // global, class
    const [globalPercentage, setGlobalPercentage] = useState(20);
    const [classPercentages, setClassPercentages] = useState({});
    const [classes, setClasses] = useState([]);
    const [loadingClasses, setLoadingClasses] = useState(false);
    const [seed, setSeed] = useState(42);

    useEffect(() => {
        if (sourcePath) {
            fetchClasses(sourcePath);
        }
    }, [sourcePath]);

    const fetchClasses = async (path) => {
        setLoadingClasses(true);
        try {
            const res = await fetch(`http://localhost:8000/api/dataset/classes?path=${encodeURIComponent(path)}`);
            if (res.ok) {
                const data = await res.json();
                setClasses(data.classes);

                // Initialize class percentages
                const initial = {};
                data.classes.forEach((c, idx) => initial[idx] = 20);
                setClassPercentages(initial);
            } else {
                setClasses([]);
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoadingClasses(false);
        }
    };

    const handleClassPercentChange = (idx, val) => {
        setClassPercentages(prev => ({
            ...prev,
            [idx]: parseFloat(val)
        }));
    };

    const handleRunSampling = async () => {
        if (!sourcePath) return alert("Please select a dataset path");

        const payload = {
            dataset_path: sourcePath,
            output_path: outputPath || null,
            seed: parseInt(seed),
            percentage: samplingMode === 'global' ? parseFloat(globalPercentage) : null,
            class_percentages: samplingMode === 'class' ? JSON.stringify(classPercentages) : null
        };

        try {
            const res = await fetch('http://localhost:8000/api/dataset/sample_task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                onStartTask();
                showNotification("Sampling started...");
            } else {
                const err = await res.json();
                alert("Error: " + err.detail);
            }
        } catch (e) {
            alert("Error running sampling: " + e.message);
        }
    };

    return (
        <div className="glass" style={{ padding: '20px', borderRadius: '16px' }}>
            <h3 style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ fontSize: '1.5rem' }}>🎲</span> Dataset Sampling
            </h3>

            <div style={{ marginBottom: '20px' }}>
                <label className="label">Source Dataset Path (Split or Flat)</label>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <input
                        className="input"
                        value={sourcePath}
                        onChange={e => setSourcePath(e.target.value)}
                        placeholder="/path/to/dataset"
                    />
                    <button className="btn" onClick={() => onBrowse(setSourcePath, 'dir')}>Browse</button>
                    <button className="btn btn-secondary" onClick={() => fetchClasses(sourcePath)} disabled={loadingClasses}>
                        {loadingClasses ? '...' : 'Fetch Classes'}
                    </button>
                </div>
            </div>

            <div style={{ marginBottom: '20px' }}>
                <label className="label">Output Path (Optional)</label>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <input
                        className="input"
                        value={outputPath}
                        onChange={e => setOutputPath(e.target.value)}
                        placeholder="Leave empty to auto-generate"
                    />
                    <button className="btn" onClick={() => onBrowse(setOutputPath, 'dir')}>Browse</button>
                </div>
            </div>

            <div style={{ marginBottom: '20px', display: 'flex', gap: '20px', alignItems: 'center' }}>
                <div style={{ flex: 1 }}>
                    <label className="label">Random Seed</label>
                    <input
                        type="number"
                        className="input"
                        value={seed}
                        onChange={e => setSeed(e.target.value)}
                        style={{ maxWidth: '100px' }}
                    />
                </div>
                <div style={{ flex: 2 }}>
                    <label className="label">Sampling Mode</label>
                    <div style={{ display: 'flex', gap: '10px', background: 'rgba(255,255,255,0.05)', padding: '5px', borderRadius: '8px' }}>
                        <button
                            className={`btn ${samplingMode === 'global' ? 'btn-primary' : ''}`}
                            style={{ flex: 1, border: 'none' }}
                            onClick={() => setSamplingMode('global')}
                        >
                            Global Percentage
                        </button>
                        <button
                            className={`btn ${samplingMode === 'class' ? 'btn-primary' : ''}`}
                            style={{ flex: 1, border: 'none' }}
                            onClick={() => setSamplingMode('class')}
                            disabled={classes.length === 0}
                        >
                            Per Class
                        </button>
                    </div>
                </div>
            </div>

            {samplingMode === 'global' && (
                <div style={{ marginBottom: '20px', background: 'rgba(0,0,0,0.2)', padding: '15px', borderRadius: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                        <label>Percentage ({globalPercentage}%)</label>
                    </div>
                    <input
                        type="range"
                        min="1" max="100"
                        value={globalPercentage}
                        onChange={e => setGlobalPercentage(e.target.value)}
                        style={{ width: '100%', accentColor: 'var(--primary)' }}
                    />
                </div>
            )}

            {samplingMode === 'class' && (
                <div style={{ marginBottom: '20px', maxHeight: '300px', overflowY: 'auto', background: 'rgba(0,0,0,0.2)', padding: '15px', borderRadius: '8px' }}>
                    <label className="label" style={{ marginBottom: '10px', display: 'block' }}>Per-Class Percentages</label>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '15px' }}>
                        {classes.map((cls, idx) => (
                            <div key={idx} style={{ padding: '10px', background: 'rgba(255,255,255,0.05)', borderRadius: '6px' }}>
                                <div style={{ fontSize: '0.9rem', marginBottom: '5px', fontWeight: 'bold' }}>{cls}</div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                    <input
                                        type="range"
                                        min="0" max="100"
                                        value={classPercentages[idx] || 0}
                                        onChange={e => handleClassPercentChange(idx, e.target.value)}
                                        style={{ flex: 1, accentColor: 'var(--primary)' }}
                                    />
                                    <span style={{ fontSize: '0.8rem', width: '35px', textAlign: 'right' }}>{classPercentages[idx] || 0}%</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div style={{ marginTop: '30px' }}>
                {!isTaskRunning ? (
                    <button className="btn btn-primary" style={{ width: '100%', padding: '15px', fontSize: '1.1rem' }} onClick={handleRunSampling}>
                        Start Sampling
                    </button>
                ) : (
                    <div>
                        <div style={{ textAlign: 'center', marginBottom: '10px', opacity: 0.7 }}>Task in progress...</div>
                        <ProgressBar progress={taskProgress} type="sampling" />
                    </div>
                )}
            </div>
        </div>
    );
};

export default DatasetSampling;
