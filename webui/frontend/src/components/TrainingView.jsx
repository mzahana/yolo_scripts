import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const API_BASE = '/api';

const TrainingView = ({ datasetPath, onBrowse }) => {
    const [mode, setMode] = useState('train'); // 'train' or 'export'

    // Common State
    const [models, setModels] = useState({});
    const [devices, setDevices] = useState([]);
    const [selectedFramework, setSelectedFramework] = useState('YOLOv8');
    const [status, setStatus] = useState({ status: 'idle', logs: [] });
    const [isPolling, setIsPolling] = useState(true);

    // Training State
    const [task, setTask] = useState('detect');
    const [selectedModel, setSelectedModel] = useState('yolov8n.pt');
    const [useCustomModel, setUseCustomModel] = useState(false);
    const [customModelPath, setCustomModelPath] = useState('');
    const [selectedDevices, setSelectedDevices] = useState(['']); // '' = Auto/Default
    const [localDatasetPath, setLocalDatasetPath] = useState(datasetPath || '');
    const [epochs, setEpochs] = useState(100);
    const [batch, setBatch] = useState(16);
    const [imgsz, setImgsz] = useState(640);
    const [workers, setWorkers] = useState(8);
    const [runName, setRunName] = useState('train');
    const [patience, setPatience] = useState(50);
    const [advancedParams, setAdvancedParams] = useState('');
    const [resultMountUrl, setResultMountUrl] = useState('');
    const [cacheBuster, setCacheBuster] = useState(Date.now());
    const [previewImage, setPreviewImage] = useState(null);
    const logsContainerRef = useRef(null);
    const isUserScrolledUpRef = useRef(false);

    // Export State
    const [exportModelPath, setExportModelPath] = useState('');
    const [exportFormat, setExportFormat] = useState('onnx');
    const [exportParams, setExportParams] = useState({
        imgsz: 640,
        batch: 1,
        half: false,
        int8: false,
        dynamic: false,
        simplify: false,
        opset: 12, // default useful for many
        workspace: 4, // GB
        nms: false,
        data: datasetPath || '', // needed for int8
        device: ''
    });
    const [exportStatus, setExportStatus] = useState({ status: 'idle', logs: [], output_file: null });
    const exportLogsRef = useRef(null);
    const isExportScrolledUpRef = useRef(false);


    // Load available models and devices on mount
    useEffect(() => {
        axios.get(`${API_BASE}/training/models`)
            .then(res => setModels(res.data))
            .catch(err => console.error("Failed to load models", err));

        axios.get(`${API_BASE}/training/devices`)
            .then(res => setDevices(res.data))
            .catch(err => console.error("Failed to load devices", err));
    }, []);

    // Effect to update selectedModel in training when task changes
    useEffect(() => {
        if (mode === 'train' && models[selectedFramework] && models[selectedFramework].length > 0) {
            const getBase = (full) => {
                let base = full.replace('.pt', '');
                base = base.replace(/-seg|-cls|-obb|-pose/g, '');
                return base;
            };

            const currentBase = getBase(selectedModel || models[selectedFramework][0]);
            const isValidBase = models[selectedFramework].includes(currentBase);
            const baseToUse = isValidBase ? currentBase : models[selectedFramework][0];

            let suffix = '.pt';
            if (task === 'segment') suffix = '-seg.pt';
            else if (task === 'classify') suffix = '-cls.pt';
            else if (task === 'obb') suffix = '-obb.pt';
            else if (task === 'pose') suffix = '-pose.pt';

            setSelectedModel(`${baseToUse}${suffix}`);
        }
    }, [task, selectedFramework, models, mode]);

    // Poll status (SHARED polling loop logic but separate endpoints)
    useEffect(() => {
        let interval;
        if (isPolling) {
            interval = setInterval(async () => {
                if (mode === 'train') {
                    try {
                        const res = await axios.get(`${API_BASE}/training/status`);
                        setStatus(res.data);

                        if (res.data.result_dir && !resultMountUrl) {
                            try {
                                const mRes = await axios.post(`${API_BASE}/mount?name=training_results&path=${encodeURIComponent(res.data.result_dir)}`);
                                setResultMountUrl(mRes.data.url);
                                setCacheBuster(Date.now());
                            } catch (e) { console.error("Mount error", e); }
                        }
                    } catch (err) { console.error("Training Poll error", err); }
                } else {
                    // Export polling
                    try {
                        const res = await axios.get(`${API_BASE}/export/status`);
                        setExportStatus(res.data);
                    } catch (err) { console.error("Export Poll error", err); }
                }
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [isPolling, mode, resultMountUrl]);

    // Smart Auto-Scroll for Training Logs
    const handleLogScroll = (e) => {
        const { scrollTop, scrollHeight, clientHeight } = e.target;
        if (scrollHeight - scrollTop - clientHeight > 50) {
            isUserScrolledUpRef.current = true;
        } else {
            isUserScrolledUpRef.current = false;
        }
    };
    useEffect(() => {
        if (logsContainerRef.current && !isUserScrolledUpRef.current) {
            logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
        }
    }, [status.logs]);

    // Smart Auto-Scroll for Export Logs
    const handleExportLogScroll = (e) => {
        const { scrollTop, scrollHeight, clientHeight } = e.target;
        if (scrollHeight - scrollTop - clientHeight > 50) {
            isExportScrolledUpRef.current = true;
        } else {
            isExportScrolledUpRef.current = false;
        }
    };
    useEffect(() => {
        if (exportLogsRef.current && !isExportScrolledUpRef.current) {
            exportLogsRef.current.scrollTop = exportLogsRef.current.scrollHeight;
        }
    }, [exportStatus.logs]);


    // Lightbox escape
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') setPreviewImage(null);
        };
        if (previewImage) {
            window.addEventListener('keydown', handleKeyDown);
        }
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [previewImage]);


    // --- Handlers ---

    const handleStartTrain = async () => {
        const finalDevice = selectedDevices.includes('') ? '' : selectedDevices.join(',');
        const config = {
            model: useCustomModel ? customModelPath : selectedModel,
            data: localDatasetPath && (localDatasetPath.endsWith('.yaml') || localDatasetPath.endsWith('.yml'))
                ? localDatasetPath
                : `${localDatasetPath}/data.yaml`,
            epochs: parseInt(epochs),
            batch: parseInt(batch),
            imgsz: parseInt(imgsz),
            device: finalDevice,
            workers: parseInt(workers),
            project: localDatasetPath ? `${localDatasetPath}/runs/${task}` : `runs/${task}`,
            name: runName,
            patience: parseInt(patience),
            task: task
        };

        if (advancedParams.trim()) {
            try {
                const overrides = JSON.parse(advancedParams);
                config.overrides = overrides;
            } catch (e) {
                alert("Invalid JSON in Advanced Params");
                return;
            }
        }

        try {
            await axios.post(`${API_BASE}/training/start`, config);
            setStatus(prev => ({ ...prev, status: 'running' }));
        } catch (err) {
            alert("Failed to start training: " + (err.response?.data?.detail || err.message));
        }
    };

    const handleStopTrain = async () => {
        if (!confirm("Stop training?")) return;
        try { await axios.post(`${API_BASE}/training/stop`); }
        catch (err) { alert("Failed to stop: " + err.message); }
    };

    const handleStartExport = async () => {
        if (!exportModelPath) {
            alert("Please select a model .pt file first");
            return;
        }

        try {
            const config = {
                model: exportModelPath,
                format: exportFormat,
                ...exportParams,
                imgsz: parseInt(exportParams.imgsz),
                batch: parseInt(exportParams.batch),
                workspace: parseInt(exportParams.workspace),
                opset: parseInt(exportParams.opset) || undefined
            };

            await axios.post(`${API_BASE}/export/start`, config);
            setExportStatus(prev => ({ ...prev, status: 'running' }));
        } catch (err) {
            alert("Failed to start export: " + (err.response?.data?.detail || err.message));
        }
    };

    const handleStopExport = async () => {
        if (!confirm("Stop export?")) return;
        try { await axios.post(`${API_BASE}/export/stop`); }
        catch (err) { alert("Failed to stop: " + err.message); }
    };


    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '20px', overflowY: 'auto', paddingRight: '10px' }}>

            {/* Mode Toggle */}
            <div style={{ display: 'flex', justifyContent: 'center', background: 'rgba(0,0,0,0.3)', padding: '5px', borderRadius: '8px' }}>
                <button
                    className={`btn ${mode === 'train' ? 'btn-primary' : ''}`}
                    style={{ flex: 1, borderRadius: '6px' }}
                    onClick={() => setMode('train')}
                >
                    Train Model
                </button>
                <div style={{ width: '10px' }}></div>
                <button
                    className={`btn ${mode === 'export' ? 'btn-primary' : ''}`}
                    style={{ flex: 1, borderRadius: '6px' }}
                    onClick={() => setMode('export')}
                >
                    Export Model
                </button>
            </div>

            {/* TRAIN MODE UI */}
            {mode === 'train' && (
                <>
                    <div className="glass" style={{ padding: '20px', borderRadius: '12px' }}>
                        <h2 style={{ marginTop: 0, borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '10px' }}>Training Configuration</h2>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                            {/* ... (Existing Training Fields) ... */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                                <div>
                                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Dataset YAML</label>
                                    <div style={{ display: 'flex', gap: '10px' }}>
                                        <input className="input" style={{ flex: 1 }} value={localDatasetPath} onChange={(e) => setLocalDatasetPath(e.target.value)} placeholder="Select dataset..." />
                                        <button className="btn btn-primary" onClick={() => onBrowse('dataset', 'dir', setLocalDatasetPath)}>Browse</button>
                                    </div>
                                    <small style={{ opacity: 0.6 }}>Point to folder with data.yaml</small>
                                </div>
                                <div>
                                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Task</label>
                                    <select className="input" style={{ width: '100%' }} value={task} onChange={e => setTask(e.target.value)}>
                                        <option value="detect">Detect</option>
                                        <option value="segment">Segment</option>
                                        <option value="classify">Classify</option>
                                        <option value="obb">OBB</option>
                                        <option value="pose">Pose</option>
                                    </select>
                                </div>
                                <div>
                                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Model</label>
                                    <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                                        <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}>
                                            <input type="radio" checked={!useCustomModel} onChange={() => setUseCustomModel(false)} /> Ultralytics
                                        </label>
                                        <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}>
                                            <input type="radio" checked={useCustomModel} onChange={() => setUseCustomModel(true)} /> Custom .pt
                                        </label>
                                    </div>
                                    {!useCustomModel ? (
                                        <div style={{ display: 'flex', gap: '10px' }}>
                                            <select className="input" value={selectedFramework} onChange={e => { setSelectedFramework(e.target.value); if (models[e.target.value]) setSelectedModel(models[e.target.value][0]); }} style={{ width: '120px' }}>
                                                {Object.keys(models).map(k => <option key={k} value={k}>{k}</option>)}
                                            </select>
                                            <select className="input" value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ flex: 1 }}>
                                                {models[selectedFramework]?.map(m => {
                                                    let suffix = '.pt';
                                                    let display = m;
                                                    if (task === 'segment') { display = `${m}-seg`; suffix = '-seg.pt'; }
                                                    else if (task === 'classify') { display = `${m}-cls`; suffix = '-cls.pt'; }
                                                    else if (task === 'obb') { display = `${m}-obb`; suffix = '-obb.pt'; }
                                                    else if (task === 'pose') { display = `${m}-pose`; suffix = '-pose.pt'; }
                                                    return <option key={m} value={`${m}${suffix}`}>{display}</option>
                                                })}
                                            </select>
                                        </div>
                                    ) : (
                                        <div style={{ display: 'flex', gap: '10px' }}>
                                            <input className="input" style={{ flex: 1 }} value={customModelPath} onChange={e => setCustomModelPath(e.target.value)} placeholder="/path/to/custom.pt" />
                                            <button className="btn btn-primary" onClick={() => onBrowse('model', 'file', setCustomModelPath)}>Browse</button>
                                        </div>
                                    )}
                                </div>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                                <div><label>Epochs</label><input type="number" className="input" value={epochs} onChange={e => setEpochs(e.target.value)} /></div>
                                <div><label>Batch</label><input type="number" className="input" value={batch} onChange={e => setBatch(e.target.value)} /></div>
                                <div><label>Size</label><input type="number" className="input" value={imgsz} onChange={e => setImgsz(e.target.value)} /></div>
                                <div><label>Patience</label><input type="number" className="input" value={patience} onChange={e => setPatience(e.target.value)} /></div>
                                <div><label>Run Name</label><input className="input" value={runName} onChange={e => setRunName(e.target.value)} /></div>
                                <div><label>Workers</label><input type="number" className="input" value={workers} onChange={e => setWorkers(e.target.value)} /></div>
                                <div style={{ gridColumn: 'span 2' }}>
                                    <label>Device</label>
                                    <div style={{ maxHeight: '100px', overflowY: 'auto', background: 'rgba(0,0,0,0.2)', padding: '5px', borderRadius: '4px' }}>
                                        <label style={{ display: 'block', cursor: 'pointer', fontSize: '0.85rem' }}>
                                            <input type="checkbox" checked={selectedDevices.includes('')} onChange={() => setSelectedDevices([''])} /> Auto
                                        </label>
                                        {devices.map(d => (
                                            <label key={d.id} style={{ display: 'block', cursor: 'pointer', fontSize: '0.85rem' }}>
                                                <input type="checkbox" checked={selectedDevices.includes(d.id)} onChange={(e) => {
                                                    if (e.target.checked) setSelectedDevices(prev => [...prev.filter(x => x !== ''), d.id]);
                                                    else setSelectedDevices(prev => { const next = prev.filter(x => x !== d.id); return next.length === 0 ? [''] : next; });
                                                }} /> {d.name}
                                            </label>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div style={{ marginTop: '20px' }}>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '5px', cursor: 'pointer' }} onClick={() => {
                                const el = document.getElementById('adv-params'); if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
                            }}>
                                <span style={{ fontWeight: 'bold' }}>▶ Advanced Parameters</span>
                            </label>
                            <div id="adv-params" style={{ display: 'none' }}>
                                <textarea className="input" rows={3} style={{ width: '100%', fontFamily: 'monospace' }} value={advancedParams} onChange={e => setAdvancedParams(e.target.value)} placeholder='{"optimizer": "Adam"}' />
                            </div>
                        </div>

                        <div style={{ marginTop: '20px', display: 'flex', gap: '10px' }}>
                            {status.status === 'running' ? (
                                <button className="btn" style={{ background: '#ef4444' }} onClick={handleStopTrain}>Stop Training</button>
                            ) : (
                                <button className="btn btn-primary" onClick={handleStartTrain} disabled={status.status === 'stopping'}>{status.status === 'stopping' ? 'Stopping...' : 'Start Training'}</button>
                            )}
                            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <span>Status: </span>
                                <span style={{ fontWeight: 'bold', color: status.status === 'running' ? '#10b981' : status.status === 'error' ? '#ef4444' : status.status === 'completed' ? '#3b82f6' : 'white' }}>{status.status.toUpperCase()}</span>
                            </div>
                        </div>
                    </div>

                    {/* Logs and Results (Same as before) */}
                    <div className="glass" style={{ flex: 1, display: 'flex', flexDirection: 'column', borderRadius: '12px', overflow: 'hidden', minHeight: '300px' }}>
                        <div style={{ padding: '10px 20px', borderBottom: '1px solid rgba(255,255,255,0.1)', background: 'rgba(0,0,0,0.2)' }}><h3 style={{ margin: 0 }}>Training Logs</h3></div>
                        <div ref={logsContainerRef} onScroll={handleLogScroll} style={{ flex: 1, overflowY: 'auto', padding: '15px', background: '#0f172a', fontFamily: 'monospace', fontSize: '0.9rem', color: '#e2e8f0' }}>
                            {status.logs.length === 0 && <div style={{ opacity: 0.5 }}>Waiting for logs...</div>}
                            {status.logs.map((line, i) => <div key={i} style={{ whiteSpace: 'pre-wrap', marginBottom: '2px' }}>{line}</div>)}
                        </div>
                    </div>

                    {(resultMountUrl || status.result_summary) && (
                        <div className="glass" style={{ padding: '20px', borderRadius: '12px' }}>
                            <h3 style={{ marginTop: 0 }}>Results</h3>
                            {status.result_summary && (
                                <div style={{ marginBottom: '20px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '10px' }}>
                                    {Object.entries(status.result_summary).map(([key, val]) => (
                                        <div key={key} style={{ background: 'rgba(0,0,0,0.3)', padding: '10px', borderRadius: '8px', textAlign: 'center' }}>
                                            <div style={{ fontSize: '0.75rem', opacity: 0.7, textTransform: 'uppercase' }}>{key.replace('_', ' ')}</div>
                                            <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#10b981' }}>{typeof val === 'number' ? (val < 1 && val > 0 ? val.toFixed(4) : val.toFixed(2)) : val}</div>
                                        </div>
                                    ))}
                                </div>
                            )}
                            {resultMountUrl && (
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '20px' }}>
                                    {['results', 'confusion_matrix', 'confusion_matrix_normalized'].map(name => (
                                        <div key={name}>
                                            <div style={{ marginBottom: '5px', fontWeight: 'bold', fontSize: '0.9rem' }}>{name.replace(/_/g, ' ')}</div>
                                            <img
                                                src={`${resultMountUrl}/${name}.png?t=${cacheBuster}`}
                                                alt={name}
                                                style={{ width: '100%', borderRadius: '8px', border: '1px solid #333', cursor: 'pointer' }}
                                                onClick={() => setPreviewImage(`${resultMountUrl}/${name}.png?t=${cacheBuster}`)}
                                                onError={(e) => e.target.style.display = 'none'}
                                            />
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </>
            )}

            {/* EXPORT MODE UI */}
            {mode === 'export' && (
                <>
                    <div className="glass" style={{ padding: '20px', borderRadius: '12px' }}>
                        <h2 style={{ marginTop: 0, borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '10px' }}>Export Configuration</h2>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                                <div>
                                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Trained Model (.pt)</label>
                                    <div style={{ display: 'flex', gap: '10px' }}>
                                        <input className="input" style={{ flex: 1 }} value={exportModelPath} onChange={(e) => setExportModelPath(e.target.value)} placeholder="/path/to/best.pt" />
                                        <button className="btn btn-primary" onClick={() => onBrowse('model', 'file', setExportModelPath)}>Browse</button>
                                    </div>
                                </div>
                                <div>
                                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Export Format</label>
                                    <select className="input" style={{ width: '100%' }} value={exportFormat} onChange={e => setExportFormat(e.target.value)}>
                                        <option value="onnx">ONNX</option>
                                        <option value="engine">TensorRT (engine)</option>
                                        <option value="openvino">OpenVINO</option>
                                        <option value="torchscript">TorchScript</option>
                                        <option value="coreml">CoreML</option>
                                        <option value="saved_model">TF SavedModel</option>
                                        <option value="pb">TF GraphDef</option>
                                        <option value="tflite">TF Lite</option>
                                        <option value="edgetpu">TF Edge TPU</option>
                                        <option value="tfjs">TF.js</option>
                                        <option value="paddle">PaddlePaddle</option>
                                        <option value="ncnn">NCNN</option>
                                    </select>
                                </div>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                                    <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}>
                                        <input type="checkbox" checked={exportParams.half} onChange={e => setExportParams(p => ({ ...p, half: e.target.checked }))} /> FP16 (Half)
                                    </label>
                                    <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}>
                                        <input type="checkbox" checked={exportParams.int8} onChange={e => setExportParams(p => ({ ...p, int8: e.target.checked }))} /> INT8 Quantization
                                    </label>
                                    <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}>
                                        <input type="checkbox" checked={exportParams.dynamic} onChange={e => setExportParams(p => ({ ...p, dynamic: e.target.checked }))} /> Dynamic Axes
                                    </label>
                                    <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}>
                                        <input type="checkbox" checked={exportParams.simplify} onChange={e => setExportParams(p => ({ ...p, simplify: e.target.checked }))} /> Simplify (ONNX)
                                    </label>
                                    <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}>
                                        <input type="checkbox" checked={exportParams.nms} onChange={e => setExportParams(p => ({ ...p, nms: e.target.checked }))} /> Add NMS
                                    </label>
                                </div>
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', alignContent: 'start' }}>
                                <div><label>Image Size</label><input type="number" className="input" value={exportParams.imgsz} onChange={e => setExportParams(p => ({ ...p, imgsz: e.target.value }))} /></div>
                                <div><label>Batch Size</label><input type="number" className="input" value={exportParams.batch} onChange={e => setExportParams(p => ({ ...p, batch: e.target.value }))} /></div>
                                <div><label>Workspace (GB)</label><input type="number" className="input" value={exportParams.workspace} onChange={e => setExportParams(p => ({ ...p, workspace: e.target.value }))} /></div>
                                <div><label>Opset</label><input type="number" className="input" value={exportParams.opset} onChange={e => setExportParams(p => ({ ...p, opset: e.target.value }))} /></div>
                                <div style={{ gridColumn: 'span 2' }}>
                                    <label>Calibration Data (YAML) - Required for INT8</label>
                                    <div style={{ display: 'flex', gap: '10px' }}>
                                        <input className="input" style={{ flex: 1 }} value={exportParams.data} onChange={e => setExportParams(p => ({ ...p, data: e.target.value }))} placeholder="data.yaml..." />
                                        <button className="btn btn-primary" onClick={() => onBrowse('dataset', 'dir', (val) => setExportParams(p => ({ ...p, data: val })))}>Browse</button>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div style={{ marginTop: '20px', display: 'flex', gap: '10px' }}>
                            {exportStatus.status === 'running' ? (
                                <button className="btn" style={{ background: '#ef4444' }} onClick={handleStopExport}>Stop Export</button>
                            ) : (
                                <button className="btn btn-primary" onClick={handleStartExport} disabled={exportStatus.status === 'stopping'}>{exportStatus.status === 'stopping' ? 'Stopping...' : 'Start Export'}</button>
                            )}
                            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <span>Status: </span>
                                <span style={{ fontWeight: 'bold', color: exportStatus.status === 'running' ? '#10b981' : exportStatus.status === 'error' ? '#ef4444' : exportStatus.status === 'completed' ? '#3b82f6' : 'white' }}>{exportStatus.status.toUpperCase()}</span>
                            </div>
                        </div>
                    </div>

                    {/* Exprot Logs */}
                    <div className="glass" style={{ flex: 1, display: 'flex', flexDirection: 'column', borderRadius: '12px', overflow: 'hidden', minHeight: '300px' }}>
                        <div style={{ padding: '10px 20px', borderBottom: '1px solid rgba(255,255,255,0.1)', background: 'rgba(0,0,0,0.2)' }}><h3 style={{ margin: 0 }}>Export Logs</h3></div>
                        <div ref={exportLogsRef} onScroll={handleExportLogScroll} style={{ flex: 1, overflowY: 'auto', padding: '15px', background: '#0f172a', fontFamily: 'monospace', fontSize: '0.9rem', color: '#e2e8f0' }}>
                            {exportStatus.logs.length === 0 && <div style={{ opacity: 0.5 }}>Waiting for logs...</div>}
                            {exportStatus.logs.map((line, i) => <div key={i} style={{ whiteSpace: 'pre-wrap', marginBottom: '2px' }}>{line}</div>)}
                        </div>
                    </div>
                </>
            )}

            {/* Lightbox Overlay */}
            {previewImage && (
                <div style={{ position: 'fixed', inset: 0, zIndex: 1000, background: 'rgba(0,0,0,0.85)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }} onClick={() => setPreviewImage(null)}>
                    <div style={{ position: 'relative', maxWidth: '95%', maxHeight: '95%' }}>
                        <img src={previewImage} style={{ maxWidth: '100%', maxHeight: '90vh', borderRadius: '8px', boxShadow: '0 0 20px rgba(0,0,0,0.5)' }} onClick={e => e.stopPropagation()} />
                        <button style={{ position: 'absolute', top: '-15px', right: '-15px', background: 'red', color: 'white', border: 'none', borderRadius: '50%', width: '30px', height: '30px', cursor: 'pointer', fontWeight: 'bold', boxShadow: '0 2px 5px rgba(0,0,0,0.3)' }} onClick={() => setPreviewImage(null)}>✕</button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default TrainingView;
