
import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_BASE = '/api';

const DEFAULT_CONFIG = {
    horizontal_flip: { enabled: false, p: 0.5 },
    vertical_flip: { enabled: false, p: 0.5 },
    rotate: { enabled: false, limit: 15, p: 0.5 },
    shear: { enabled: false, shear: 15, p: 0.5 },
    brightness: { enabled: false, brightness_limit: 0.2, contrast_limit: 0.2, p: 0.5 },
    blur: { enabled: false, blur_limit: 7, p: 0.5 },
    noise: { enabled: false, var_limit: [10.0, 50.0], p: 0.5 },
    grayscale: { enabled: false, p: 0.5 }
};

const SimpleAugmentationTool = ({
    initialDatasetPath,
    handleLandingBrowse,
    showNotification,
    isTaskRunning
}) => {
    const [datasetPath, setDatasetPath] = useState(initialDatasetPath || '');
    const [config, setConfig] = useState(DEFAULT_CONFIG);
    const [multiplier, setMultiplier] = useState(1);
    const [outputName, setOutputName] = useState('');
    const [previewData, setPreviewData] = useState(null);
    const [loadingPreview, setLoadingPreview] = useState(false);
    const [processing, setProcessing] = useState(false);
    const [progress, setProgress] = useState(null);
    const [datasetInfo, setDatasetInfo] = useState(null);
    const [selectedSplits, setSelectedSplits] = useState(['train', 'valid', 'test']);

    // Initial output name default
    useEffect(() => {
        if (datasetPath && !outputName) {
            const name = datasetPath.split('/').pop() || 'dataset';
            setOutputName(`${name}_aug`);
        }
    }, [datasetPath]);

    // Update local dataset path if parent prop changes, but only if local is empty or we want to sync
    useEffect(() => {
        if (initialDatasetPath && !datasetPath) {
            setDatasetPath(initialDatasetPath);
        }
    }, [initialDatasetPath]);

    // Fetch dataset info when path changes
    useEffect(() => {
        const fetchInfo = async () => {
            if (!datasetPath) return;
            try {
                const res = await axios.post(`${API_BASE}/augment/simple/scan`, { dataset_path: datasetPath });
                setDatasetInfo(res.data);
                if (res.data.type === 'split') {
                    setSelectedSplits(res.data.splits);
                }
            } catch (err) {
                console.error("Error scanning dataset:", err);
                // Don't error blocking, just ignore info
                setDatasetInfo(null);
            }
        };
        fetchInfo();
    }, [datasetPath]);


    const updateConfig = (key, updates) => {
        setConfig(prev => ({
            ...prev,
            [key]: { ...prev[key], ...updates }
        }));
    };

    const fetchPreview = useCallback(async () => {
        if (!datasetPath) {
            showNotification("Please select a dataset first", "warning");
            return;
        }
        setLoadingPreview(true);
        try {
            const res = await axios.post(`${API_BASE}/augment/simple/preview`, {
                dataset_path: datasetPath,
                config: config
            });
            setPreviewData(res.data);
        } catch (err) {
            console.error("Preview error:", err);
            showNotification("Error generating preview: " + (err.response?.data?.detail || err.message), "error");
        } finally {
            setLoadingPreview(false);
        }
    }, [datasetPath, config, showNotification]);

    // Initial preview if path is set
    useEffect(() => {
        if (datasetPath) fetchPreview();
    }, [datasetPath]);

    const handleRun = async () => {
        if (!datasetPath || !outputName) {
            showNotification("Please select a dataset and output name", "warning");
            return;
        }
        if (processing) return;

        setProcessing(true);
        try {
            const payload = {
                dataset_path: datasetPath,
                output_name: outputName,
                multiplier: parseInt(multiplier),
                config: config
            };

            if (datasetInfo && datasetInfo.type === 'split') {
                payload.selected_splits = selectedSplits;
            }

            const res = await axios.post(`${API_BASE}/augment/simple/run`, payload);

            showNotification(`Started! Output will be at: ${res.data.output_path}`, "success");

        } catch (err) {
            console.error("Run error:", err);
            showNotification("Error starting job: " + (err.response?.data?.detail || err.message), "error");
            setProcessing(false);
        }
    };

    // Poll progress
    useEffect(() => {
        let interval;
        if (processing) {
            interval = setInterval(async () => {
                try {
                    const res = await axios.get(`${API_BASE}/progress`);
                    const data = res.data;
                    setProgress(data);
                    if (data.status === 'idle' || data.status === 'error') {
                        setProcessing(false);
                        if (data.status === 'error') {
                            showNotification("Job failed: " + data.message, "error");
                        }
                        else if (data.status === 'idle' && data.message.includes('Finished')) {
                            showNotification("Augmentation Complete! " + data.message, "success");
                        }
                    }
                } catch (e) {
                    // ignore
                }
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [processing, showNotification]);

    // Calculate approx total images
    const getTotalImages = () => {
        if (!datasetInfo) return 0;
        if (datasetInfo.type === 'flat') return datasetInfo.total_images;

        let total = 0;
        selectedSplits.forEach(s => {
            total += (datasetInfo.split_counts[s] || 0);
        });
        return total;
    };

    const approxTotal = getTotalImages() * parseInt(multiplier);

    return (
        <div className="simple-aug-tool" style={{ height: '100%', overflowY: 'auto' }}>
            <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>

                {/* Configuration Panel */}
                <div className="glass-panel" style={{ flex: '0 0 350px', padding: '20px' }}>
                    <h3 style={{ marginTop: 0 }}>Configuration</h3>

                    {/* Dataset Selection */}
                    <div style={{ marginBottom: '20px', paddingBottom: '20px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                        <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>Input Dataset</label>
                        <div style={{ display: 'flex', gap: '10px' }}>
                            <button
                                className="btn-browse"
                                onClick={() => handleLandingBrowse(setDatasetPath, 'dir')}
                                title="Select Dataset Folder"
                            >
                                📂
                            </button>
                            <input
                                type="text"
                                value={datasetPath}
                                readOnly
                                placeholder="Select dataset..."
                                style={{
                                    flex: 1,
                                    background: 'rgba(0,0,0,0.2)',
                                    border: '1px solid rgba(255,255,255,0.1)',
                                    color: 'white',
                                    padding: '6px 10px',
                                    borderRadius: '4px'
                                }}
                            />
                        </div>
                        {datasetInfo && (
                            <div style={{ marginTop: 10, fontSize: '0.9rem', opacity: 0.8 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                                    <span style={{ fontWeight: 'bold', color: datasetInfo.type === 'split' ? '#a5b4fc' : '#4ade80' }}>
                                        {datasetInfo.type === 'split' ? 'Split Dataset' : 'Flat Dataset'}
                                    </span>
                                    <span>• {datasetInfo.total_images} images found</span>
                                </div>
                                {datasetInfo.type === 'split' && (
                                    <div style={{ marginTop: 8, background: 'rgba(0,0,0,0.1)', padding: 8, borderRadius: 4 }}>
                                        <label style={{ fontSize: '0.8rem', display: 'block', marginBottom: 5, opacity: 0.7 }}>Select Splits to Augment:</label>
                                        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                                            {datasetInfo.splits.map(split => (
                                                <label key={split} style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '0.85rem' }}>
                                                    <input
                                                        type="checkbox"
                                                        checked={selectedSplits.includes(split)}
                                                        onChange={e => {
                                                            if (e.target.checked) setSelectedSplits(p => [...p, split]);
                                                            else setSelectedSplits(p => p.filter(s => s !== split));
                                                        }}
                                                        style={{ marginRight: 4 }}
                                                    />
                                                    {split} ({datasetInfo.split_counts[split]})
                                                </label>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    <div className="aug-controls">
                        {/* Geometry */}
                        <h4>Geometry</h4>
                        <ControlRow label="Horizontal Flip"
                            enabled={config.horizontal_flip.enabled}
                            onToggle={v => updateConfig('horizontal_flip', { enabled: v })}>
                            <label>Prob: <input type="number" step="0.1" max="1" min="0" value={config.horizontal_flip.p} onChange={e => updateConfig('horizontal_flip', { p: parseFloat(e.target.value) })} /></label>
                        </ControlRow>

                        <ControlRow label="Vertical Flip"
                            enabled={config.vertical_flip.enabled}
                            onToggle={v => updateConfig('vertical_flip', { enabled: v })}>
                            <label>Prob: <input type="number" step="0.1" max="1" min="0" value={config.vertical_flip.p} onChange={e => updateConfig('vertical_flip', { p: parseFloat(e.target.value) })} /></label>
                        </ControlRow>

                        <ControlRow label="Rotate"
                            enabled={config.rotate.enabled}
                            onToggle={v => updateConfig('rotate', { enabled: v })}>
                            <div style={{ marginBottom: 5 }}>
                                <label>Limit (°): <input type="number" value={config.rotate.limit} onChange={e => updateConfig('rotate', { limit: parseInt(e.target.value) })} /></label>
                            </div>
                            <label>Prob: <input type="number" step="0.1" max="1" min="0" value={config.rotate.p} onChange={e => updateConfig('rotate', { p: parseFloat(e.target.value) })} /></label>
                        </ControlRow>

                        <ControlRow label="Shear"
                            enabled={config.shear.enabled}
                            onToggle={v => updateConfig('shear', { enabled: v })}>
                            <div style={{ marginBottom: 5 }}>
                                <label>Shear Angle: <input type="number" value={config.shear.shear} onChange={e => updateConfig('shear', { shear: parseInt(e.target.value) })} /></label>
                            </div>
                            <label>Prob: <input type="number" step="0.1" className="prob-input" value={config.shear.p} onChange={e => updateConfig('shear', { p: parseFloat(e.target.value) })} /></label>
                        </ControlRow>

                        {/* Color / Visual */}
                        <h4>Visual</h4>
                        <ControlRow label="Brightness/Contrast"
                            enabled={config.brightness.enabled}
                            onToggle={v => updateConfig('brightness', { enabled: v })}>
                            <div style={{ marginBottom: 5 }}>
                                <label>Bright Limit: <input type="number" step="0.1" value={config.brightness.brightness_limit} onChange={e => updateConfig('brightness', { brightness_limit: parseFloat(e.target.value) })} /></label>
                            </div>
                            <div style={{ marginBottom: 5 }}>
                                <label>Contr Limit: <input type="number" step="0.1" value={config.brightness.contrast_limit} onChange={e => updateConfig('brightness', { contrast_limit: parseFloat(e.target.value) })} /></label>
                            </div>
                            <label>Prob: <input type="number" step="0.1" value={config.brightness.p} onChange={e => updateConfig('brightness', { p: parseFloat(e.target.value) })} /></label>
                        </ControlRow>

                        <ControlRow label="Blur"
                            enabled={config.blur.enabled}
                            onToggle={v => updateConfig('blur', { enabled: v })}>
                            <div style={{ marginBottom: 5 }}>
                                <label>Limit: <input type="number" value={config.blur.blur_limit} onChange={e => updateConfig('blur', { blur_limit: parseInt(e.target.value) })} /></label>
                            </div>
                            <label>Prob: <input type="number" step="0.1" value={config.blur.p} onChange={e => updateConfig('blur', { p: parseFloat(e.target.value) })} /></label>
                        </ControlRow>

                        <ControlRow label="Noise"
                            enabled={config.noise.enabled}
                            onToggle={v => updateConfig('noise', { enabled: v })}>
                            <div style={{ marginBottom: 5 }}>
                                <label>Lower: <input type="number" value={config.noise.var_limit[0]} onChange={e => updateConfig('noise', { var_limit: [parseFloat(e.target.value), config.noise.var_limit[1]] })} /></label>
                                <label>Upper: <input type="number" value={config.noise.var_limit[1]} onChange={e => updateConfig('noise', { var_limit: [config.noise.var_limit[0], parseFloat(e.target.value)] })} /></label>
                            </div>
                            <label>Prob: <input type="number" step="0.1" value={config.noise.p} onChange={e => updateConfig('noise', { p: parseFloat(e.target.value) })} /></label>
                        </ControlRow>

                        <ControlRow label="Grayscale"
                            enabled={config.grayscale.enabled}
                            onToggle={v => updateConfig('grayscale', { enabled: v })}>
                            <label>Prob: <input type="number" step="0.1" value={config.grayscale.p} onChange={e => updateConfig('grayscale', { p: parseFloat(e.target.value) })} /></label>
                        </ControlRow>

                    </div>

                    <button className="btn primary" style={{ width: '100%', marginTop: 20 }} onClick={fetchPreview}>Update Preview</button>
                </div>

                {/* Main Content: Preview & Run */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 20 }}>

                    {/* Preview Area */}
                    <div className="glass-panel" style={{ padding: 20, minHeight: 400 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <h3 style={{ marginTop: 0 }}>Preview</h3>
                            {loadingPreview && <span>Loading...</span>}
                        </div>

                        {previewData && (
                            <div style={{ display: 'flex', gap: 20, justifyContent: 'center', flexWrap: 'wrap' }}>
                                <div style={{ flex: '1 1 400px', minWidth: 0, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                    <p style={{ fontWeight: 'bold', marginBottom: 5 }}>Original</p>
                                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', background: 'rgba(0,0,0,0.2)', borderRadius: 8, padding: 4 }}>
                                        <img src={previewData.original_image} alt="Original" style={{ maxWidth: '100%', maxHeight: '65vh', objectFit: 'contain' }} />
                                    </div>
                                    <p style={{ fontSize: '0.8em', color: '#888', marginTop: 5 }}>{previewData.sample_name}</p>
                                </div>
                                <div style={{ flex: '1 1 400px', minWidth: 0, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                    <p style={{ fontWeight: 'bold', marginBottom: 5 }}>Augmented</p>
                                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', background: 'rgba(0,0,0,0.2)', borderRadius: 8, padding: 4 }}>
                                        <img src={previewData.augmented_image} alt="Augmented" style={{ maxWidth: '100%', maxHeight: '65vh', objectFit: 'contain' }} />
                                    </div>
                                </div>
                            </div>
                        )}
                        {!previewData && !loadingPreview && <div style={{ padding: 40, textAlign: 'center', color: '#666' }}>Select settings and click Update Preview</div>}
                    </div>

                    {/* Execution Area */}
                    <div className="glass-panel" style={{ padding: 20 }}>
                        <h3 style={{ marginTop: 0 }}>Run Augmentation</h3>
                        <div style={{ display: 'flex', gap: 20, alignItems: 'center', marginBottom: 20 }}>
                            <div style={{ flex: 1 }}>
                                <label style={{ display: 'block', marginBottom: 5 }}>Output Folder Name</label>
                                <input type="text" value={outputName} onChange={e => setOutputName(e.target.value)} style={{ width: '100%', padding: 8, background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', borderRadius: 4 }} />
                                <div style={{ fontSize: '0.75rem', opacity: 0.5, marginTop: '3px' }}>
                                    (Created in parent directory of dataset)
                                </div>
                            </div>
                            <div style={{ flex: 1 }}>
                                <label style={{ display: 'block', marginBottom: 5 }}>Augmentations per Image (Multiplier)</label>
                                <input type="number" min="1" value={multiplier} onChange={e => setMultiplier(e.target.value)} style={{ width: '100%', padding: 8, background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', borderRadius: 4 }} />
                            </div>
                        </div>

                        {processing && progress && (
                            <div style={{ marginBottom: 20 }}>
                                <p>{progress.message}</p>
                                <progress value={progress.current} max={progress.total} style={{ width: '100%' }}></progress>
                                <div style={{ textAlign: 'right' }}>{progress.current} / {progress.total}</div>
                            </div>
                        )}

                        <button className="btn primary big" style={{ width: '100%', padding: '15px', fontSize: '1.1rem' }} onClick={handleRun} disabled={processing || !outputName}>
                            {processing ? 'Processing...' : `Generate Dataset (${approxTotal} Images)`}
                        </button>
                    </div>

                </div>
            </div>
            <style>{`
                .glass-panel { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; box-sizing: border-box; }
                .btn-browse { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: white; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
                 .btn.primary { background: var(--primary); color: white; border: none; padding: 10px; border-radius: 6px; cursor: pointer; font-weight: bold; }
                 .btn.primary:disabled { opacity: 0.5; cursor: not-allowed; }
            `}</style>
        </div>
    );
}

function ControlRow({ label, enabled, onToggle, children }) {
    return (
        <div style={{ marginBottom: 15, borderBottom: '1px solid #333', paddingBottom: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: enabled ? 10 : 0 }}>
                <label style={{ fontWeight: 500 }}>{label}</label>
                <input type="checkbox" checked={enabled} onChange={e => onToggle(e.target.checked)} />
            </div>
            {enabled && (
                <div style={{ paddingLeft: 10, fontSize: '0.9em' }}>
                    {children}
                </div>
            )}
        </div>
    );
}

export default SimpleAugmentationTool;
