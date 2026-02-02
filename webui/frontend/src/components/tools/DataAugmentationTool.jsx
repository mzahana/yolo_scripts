import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

import ProgressBar from '../ProgressBar';

// Available Augmentation Types
const AUGMENTATION_TYPES = [
    { id: 'rotation', label: 'Rotation', icon: '🔄', desc: 'Randomly rotate objects' },
    { id: 'blur', label: 'Blur', icon: '💧', desc: 'Apply Gaussian blur' },
    { id: 'scaling', label: 'Random Scaling', icon: '📏', desc: 'Randomly scale objects' },
    { id: 'contrast', label: 'Contrast', icon: '🌗', desc: 'Adjust brightness/contrast' },
];

const DataAugmentationTool = ({
    datasetPath,
    setDatasetPath,
    isTaskRunning,
    openFileBrowser,
    handleLandingBrowse,
    showNotification,
    taskProgress, // Receive progress prop
    setIsTaskRunning // Receiver setter to start polling
}) => {
    // ----------------------------------------------------
    // State
    // ----------------------------------------------------

    // Dataset & Setup
    const [localDatasetPath, setLocalDatasetPath] = useState(datasetPath || '');
    const [backgroundPath, setBackgroundPath] = useState('');
    const [backgroundName, setBackgroundName] = useState('');
    const [backgroundPreviewUrl, setBackgroundPreviewUrl] = useState(''); // Local preview URL
    const [stats, setStats] = useState(null); // { total_objects, class_counts, class_percentages }

    // Classes
    const [availableClasses, setAvailableClasses] = useState([]);
    const [selectedClasses, setSelectedClasses] = useState([]);

    // Pipeline (List of confirmed augmentations)
    // Structure: { id, type, params: { ... } }
    const [pipeline, setPipeline] = useState([]);

    // Editing State (Wizard)
    const [isEditing, setIsEditing] = useState(false);
    const [editingType, setEditingType] = useState(null); // 'rotation', 'blur', etc.
    const [currentParams, setCurrentParams] = useState({});

    // Sampling & Preview
    // 'originalSample' is the result of /sample (Object on background with NO augs)
    // 'previewImage' is the result of /apply_preview (Object on background WITH augs)
    const [originalSample, setOriginalSample] = useState(null);
    const [previewImage, setPreviewImage] = useState(null);
    const [isLoadingSample, setIsLoadingSample] = useState(false);
    const [isPreviewing, setIsPreviewing] = useState(false);

    // Global Params (ROI) - Always active, applied to everything
    // const [regionScale, setRegionScale] = useState(0.8); // Removed as per user request
    const [roi, setRoi] = useState({ x: 0, y: 0, w: 100, h: 100 });
    const [numAugmentations, setNumAugmentations] = useState(3);
    const [augmentTogether, setAugmentTogether] = useState(false);

    // Composition Mode State
    const [augmentationMode, setAugmentationMode] = useState('single'); // 'single', 'composition'
    const [compTotalImages, setCompTotalImages] = useState(10);
    const [compObjectsPerImage, setCompObjectsPerImage] = useState(3);
    const [customOutputName, setCustomOutputName] = useState('');


    // Interactive ROI Selection State
    const bgContainerRef = useRef(null);
    const [isSelecting, setIsSelecting] = useState(false);
    const [selectionStart, setSelectionStart] = useState(null); // {x, y} percentage

    // Dropdown State
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);

    // ----------------------------------------------------
    // Effects
    // ----------------------------------------------------
    useEffect(() => {
        if (localDatasetPath) {
            handleFetchClasses(localDatasetPath);
            fetchStats(localDatasetPath);
        }
    }, [localDatasetPath]);

    // ----------------------------------------------------
    // API Calls
    // ----------------------------------------------------

    const handleFetchClasses = async (path) => {
        try {
            const res = await axios.get(`/api/dataset/classes`, { params: { path } });
            if (res.data.classes) {
                setAvailableClasses(res.data.classes);
                // Auto-select all by default if none selected
                if (selectedClasses.length === 0) setSelectedClasses(res.data.classes.map((_, i) => i));
            }
        } catch (err) { console.error(err); }
    };

    const fetchStats = async (path) => {
        try {
            const res = await axios.get(`/api/augmentation/stats`, { params: { path } });
            setStats(res.data);
        } catch (err) { console.error(err); }
    };

    const handleBackgroundUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // precise UI preview
        const objectUrl = URL.createObjectURL(file);
        setBackgroundPreviewUrl(objectUrl);

        const formData = new FormData();
        formData.append("file", file);
        try {
            const res = await axios.post("/api/augmentation/upload_background", formData);
            setBackgroundPath(res.data.path);
            setBackgroundName(file.name);
            showNotification("Background uploaded", "success");
        } catch (err) {
            showNotification("Upload failed", "error");
        }
    };

    const drawSample = async () => {
        if (!localDatasetPath || !backgroundPath || selectedClasses.length === 0) {
            showNotification("Select dataset, background and classes first", "warning");
            return;
        }
        setIsLoadingSample(true);
        try {
            // First draw the "Original" sample (no augs)
            const res = await axios.post("/api/augmentation/sample", {
                dataset_path: localDatasetPath,
                class_ids: selectedClasses,
                background_path: backgroundPath
            });
            setOriginalSample(res.data.image);

            // If we are currently editing, we should also update the augmented preview for this new object
            if (isEditing) {
                updatePreview(currentParams);
            }
        } catch (err) {
            showNotification("Sample failed: " + (err.response?.data?.detail || err.message), "error");
        } finally {
            setIsLoadingSample(false);
        }
    };

    const updatePreview = async (params) => {
        if (!originalSample) return; // Need a sample first
        setIsPreviewing(true);
        try {
            // Check what type we are editing to construct payload
            const payload = {
                background_path: backgroundPath,
                region_scale: 1.0, // Hardcoded as redundant
                roi: [roi.x / 100, roi.y / 100, roi.w / 100, roi.h / 100],
                // Add specific params
                rotation_range: params.rotation_range || null,
                blur_range: params.blur_range || null,
                scaling_range: params.scaling_range || null,
                contrast_range: params.contrast_range || null
            };

            const res = await axios.post("/api/augmentation/apply_preview", payload);
            setPreviewImage(res.data.image);
        } catch (err) {
            console.error(err);
        } finally {
            setIsPreviewing(false);
        }
    };

    // Debounced update for sliders
    const timeoutRef = useRef(null);
    const onParamChange = (newParams) => {
        setCurrentParams(newParams);
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        timeoutRef.current = setTimeout(() => {
            updatePreview(newParams);
        }, 300);
    };


    // ----------------------------------------------------
    // Pipeline Management
    // ----------------------------------------------------

    const startAdding = (type) => {
        // Init default params based on type
        let defaults = {};
        if (type === 'rotation') defaults = { rotation_range: [-15, 15] };
        if (type === 'blur') defaults = { blur_range: [0, 3] };
        if (type === 'scaling') defaults = { scaling_range: [0.8, 1.2] };
        if (type === 'contrast') defaults = { contrast_range: [0.8, 1.2] };

        setEditingType(type);
        setCurrentParams(defaults);
        setIsEditing(true);

        setIsEditing(true);
        setIsDropdownOpen(false); // Close dropdown

        // Auto-draw sample if missing
        if (!originalSample) drawSample();
        else updatePreview(defaults); // Update preview for existing sample with new defaults
    };

    const confirmAdd = () => {
        setPipeline([...pipeline, { id: Date.now(), type: editingType, params: currentParams }]);
        setIsEditing(false);
        setEditingType(null);
        setPreviewImage(null); // Clear preview
    };

    const cancelAdd = () => {
        setIsEditing(false);
        setEditingType(null);
        setPreviewImage(null);
    };

    const removeStep = (index) => {
        const newPipe = [...pipeline];
        newPipe.splice(index, 1);
        setPipeline(newPipe);
    };

    const handleGenerate = async () => {
        if (pipeline.length === 0) {
            showNotification("Add at least one augmentation", "warning");
            return;
        }

        // Merge all pipeline params into one payload
        // Note: The current backend design is flat (one range per type).
        // If user added 2 rotation steps, we might need to merging logic or just take the last one?
        // Ideally, we should block adding duplicates if backend doesn't support stacks.
        // For now, let's merge.

        let mergedParams = {};
        pipeline.forEach(step => {
            mergedParams = { ...mergedParams, ...step.params };
        });

        const payload = {
            dataset_path: localDatasetPath,
            background_path: backgroundPath,
            class_ids: selectedClasses,
            num_augmentations: numAugmentations,
            augment_together: augmentTogether,
            region_scale: 1.0,
            roi: [roi.x / 100, roi.y / 100, roi.w / 100, roi.h / 100],
            composition_mode: augmentationMode === 'composition',
            total_images: compTotalImages,
            objects_per_image: compObjectsPerImage,
            custom_output_name: customOutputName,
            ...mergedParams
        };

        try {
            const res = await axios.post("/api/augmentation/generate", payload);
            setIsTaskRunning(true); // Start polling in App.jsx
            showNotification(`Started! Output: ${res.data.output_path}`, "success");
        } catch (err) {
            showNotification("Failed to start: " + err.message, "error");
        }
    };


    // ----------------------------------------------------
    // Interactive ROI Handlers
    // ----------------------------------------------------
    const handleMouseDown = (e) => {
        if (!bgContainerRef.current) return;
        const rect = bgContainerRef.current.getBoundingClientRect();
        // Calculate percentage coordinates
        const x = (e.clientX - rect.left) / rect.width * 100;
        const y = (e.clientY - rect.top) / rect.height * 100;

        setSelectionStart({ x, y });
        setIsSelecting(true);
        // Init zero-size box
        setRoi({ x: Math.round(x), y: Math.round(y), w: 0, h: 0 });
    };

    const handleMouseMove = (e) => {
        if (!isSelecting || !bgContainerRef.current) return;
        const rect = bgContainerRef.current.getBoundingClientRect();
        const currentX = (e.clientX - rect.left) / rect.width * 100;
        const currentY = (e.clientY - rect.top) / rect.height * 100;

        const startX = selectionStart.x;
        const startY = selectionStart.y;

        let newX = Math.min(startX, currentX);
        let newY = Math.min(startY, currentY);
        let newW = Math.abs(currentX - startX);
        let newH = Math.abs(currentY - startY);

        // Clamp to 0-100 bounds
        newX = Math.max(0, Math.min(100, newX));
        newY = Math.max(0, Math.min(100, newY));
        if (newX + newW > 100) newW = 100 - newX;
        if (newY + newH > 100) newH = 100 - newY;

        setRoi({ x: Math.round(newX), y: Math.round(newY), w: Math.round(newW), h: Math.round(newH) });
    };

    const handleMouseUp = () => {
        setIsSelecting(false);
        setSelectionStart(null);
    };

    // ----------------------------------------------------
    // Render Helpers
    // ----------------------------------------------------

    const renderRangeSlider = (label, key, min, max, step, unit = '') => {
        // Extract current range from currentParams
        const val = currentParams[key] || [min, max]; // Default if missing
        return (
            <div style={{ marginBottom: '15px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px', fontSize: '0.9rem' }}>
                    <span>{label}</span>
                    <span style={{ opacity: 0.7 }}>{val[0]} {unit} - {val[1]} {unit}</span>
                </div>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center', width: '100%' }}>
                    <input type="range" min={min} max={max} step={step} value={val[0]}
                        onChange={e => {
                            const v = parseFloat(e.target.value);
                            const newRange = [Math.min(v, val[1]), val[1]];
                            onParamChange({ ...currentParams, [key]: newRange });
                        }}
                        style={{ flex: 1, minWidth: 0, width: 0 }}
                    />
                    <input type="range" min={min} max={max} step={step} value={val[1]}
                        onChange={e => {
                            const v = parseFloat(e.target.value);
                            const newRange = [val[0], Math.max(v, val[0])];
                            onParamChange({ ...currentParams, [key]: newRange });
                        }}
                        style={{ flex: 1, minWidth: 0, width: 0 }}
                    />
                </div>
            </div>
        );
    };

    // ----------------------------------------------------
    // UI Layout
    // ----------------------------------------------------

    // 1. Selector View (Main)
    if (!isEditing) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflowY: 'auto', paddingRight: '10px' }}>

                {/* 1. Dataset & Background (Full Width Row) */}
                <div className="glass-panel" style={{ padding: '20px' }}>
                    <h4 style={{ margin: '0 0 15px 0' }}>1. Dataset & Background</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                        <div>
                            <label style={{ display: 'block', fontSize: '0.85rem', opacity: 0.8, marginBottom: '8px' }}>Source Dataset</label>
                            <div style={{ display: 'flex', gap: '10px' }}>
                                <button onClick={() => handleLandingBrowse(setLocalDatasetPath, 'dir')} className="btn-browse" style={{ whiteSpace: 'nowrap' }}>📂 Select Folder</button>
                                <div className="path-display" title={localDatasetPath}>{localDatasetPath || 'No dataset selected'}</div>
                            </div>
                        </div>
                        <div>
                            <label style={{ display: 'block', fontSize: '0.85rem', opacity: 0.8, marginBottom: '8px' }}>Background Image</label>
                            <div style={{ display: 'flex', gap: '10px' }}>
                                <label className="btn-browse" style={{ display: 'inline-block', cursor: 'pointer', whiteSpace: 'nowrap' }}>
                                    🖼️ Upload Image
                                    <input type="file" accept="image/*" onChange={handleBackgroundUpload} style={{ display: 'none' }} />
                                </label>
                                <div className="path-display" title={backgroundName}>{backgroundName || 'No background selected'}</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* 2. Classes (Full Width Row) */}
                <div className="glass-panel" style={{ padding: '20px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '15px', alignItems: 'center' }}>
                        <h4 style={{ margin: 0 }}>2. Classes <span style={{ fontSize: '0.85rem', fontWeight: 'normal', opacity: 0.7 }}>({selectedClasses.length}/{availableClasses.length})</span></h4>
                        <button
                            style={{ fontSize: '0.8rem', color: 'var(--primary)', cursor: 'pointer', background: 'rgba(255,255,255,0.05)', border: 'none', padding: '5px 15px', borderRadius: '4px' }}
                            onClick={() => setSelectedClasses(selectedClasses.length === availableClasses.length ? [] : availableClasses.map((_, i) => i))}
                        >
                            {selectedClasses.length === availableClasses.length ? 'Deselect All' : 'Select All'}
                        </button>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', maxHeight: '150px', overflowY: 'auto', padding: '5px' }}>
                        {availableClasses.length > 0 ? availableClasses.map((cls, i) => (
                            <span key={i}
                                onClick={() => {
                                    if (selectedClasses.includes(i)) setSelectedClasses(selectedClasses.filter(x => x !== i));
                                    else setSelectedClasses([...selectedClasses, i]);
                                }}
                                style={{
                                    fontSize: '0.9rem', padding: '6px 14px', borderRadius: '20px', cursor: 'pointer', border: '1px solid transparent',
                                    background: selectedClasses.includes(i) ? 'var(--primary)' : 'rgba(255,255,255,0.05)',
                                    color: selectedClasses.includes(i) ? 'white' : 'var(--text-color)',
                                    transition: 'all 0.2s',
                                    boxShadow: selectedClasses.includes(i) ? '0 2px 5px rgba(0,0,0,0.2)' : 'none'
                                }}
                            >
                                {cls}
                            </span>
                        )) : <div style={{ opacity: 0.6, padding: '10px', fontStyle: 'italic' }}>Select a dataset first to load classes</div>}
                    </div>
                </div>

                {/* 3. Stats (Full Width Row) */}
                <div className="glass-panel" style={{ padding: '20px' }}>
                    <h4 style={{ margin: '0 0 15px 0' }}>3. Dataset Statistics</h4>
                    {stats ? (
                        <div style={{ display: 'flex', gap: '30px', alignItems: 'flex-start' }}>
                            <div style={{ minWidth: '150px', padding: '20px', background: 'rgba(255,255,255,0.05)', borderRadius: '12px', textAlign: 'center' }}>
                                <div style={{ opacity: 0.7, fontSize: '0.9rem', marginBottom: '8px' }}>Total Objects</div>
                                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--primary)' }}>{stats.total_objects.toLocaleString()}</div>
                            </div>
                            <div style={{ flex: 1, maxHeight: '200px', overflowY: 'auto', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '12px' }}>
                                {Object.entries(stats.class_percentages).map(([cid, pct]) => (
                                    <div key={cid} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 15px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', alignItems: 'center' }}>
                                        <span style={{ opacity: 0.9, fontSize: '0.9rem', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', maxWidth: '140px' }} title={availableClasses[cid]}>{availableClasses[cid] || `Class ${cid}`}</span>
                                        <span style={{ opacity: 0.9, fontFamily: 'monospace', fontWeight: 'bold' }}>{pct.toFixed(1)}%</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : <div style={{ opacity: 0.6, fontStyle: 'italic', padding: '10px' }}>Select dataset to view statistics</div>}
                </div>

                {/* 4. Global Settings (Full Width Row) */}
                <div className="glass-panel" style={{ padding: '20px' }}>
                    <h4 style={{ margin: '0 0 15px 0' }}>4. Global Settings</h4>
                    <div style={{ display: 'flex', gap: '40px', flexWrap: 'nowrap', alignItems: 'flex-start' }}>

                        {/* Visualization Area */}
                        {backgroundPreviewUrl ? (
                            <div
                                ref={bgContainerRef}
                                onMouseDown={handleMouseDown}
                                onMouseMove={handleMouseMove}
                                onMouseUp={handleMouseUp}
                                onMouseLeave={handleMouseUp}
                                style={{ width: '300px', position: 'relative', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.2)', overflow: 'hidden', cursor: 'crosshair', userSelect: 'none' }}
                            >
                                <img src={backgroundPreviewUrl} alt="Background" style={{ width: '100%', display: 'block', pointerEvents: 'none' }} />
                                <div style={{
                                    position: 'absolute',
                                    left: `${roi.x}%`, top: `${roi.y}%`, width: `${roi.w}%`, height: `${roi.h}%`,
                                    border: '2px solid #00ff00', background: 'rgba(0, 255, 0, 0.2)',
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    pointerEvents: 'none' // Ensure clicks pass through if needed
                                }}>
                                    <span style={{ color: '#00ff00', fontSize: '0.7rem', fontWeight: 'bold', textShadow: '0 1px 2px black' }}>ROI</span>
                                </div>
                                <div style={{ position: 'absolute', bottom: '5px', right: '5px', fontSize: '0.7rem', background: 'rgba(0,0,0,0.6)', padding: '2px 5px', borderRadius: '4px' }}>Preview</div>
                            </div>
                        ) : (
                            <div style={{ width: '300px', height: '180px', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', border: '1px dashed rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column' }}>
                                <div style={{ fontSize: '2rem', marginBottom: '10px', opacity: 0.3 }}>🖼️</div>
                                <div style={{ opacity: 0.5, fontSize: '0.85rem' }}>Select background to visualize ROI</div>
                            </div>
                        )}

                        {/* Controls */}
                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
                            {/* ROI Inputs */}
                            <div>
                                <div style={{ fontSize: '0.9rem', opacity: 0.9, marginBottom: '10px' }}>Random Placement ROI (X, Y, W, H %)</div>
                                <div style={{ display: 'flex', gap: '15px' }}>
                                    {['x', 'y', 'w', 'h'].map(k => (
                                        <div key={k} style={{ flex: 1 }}>
                                            <div style={{ fontSize: '0.75rem', opacity: 0.6, textAlign: 'center', marginBottom: '4px' }}>{k.toUpperCase()}</div>
                                            <input type="number" value={roi[k]}
                                                onChange={e => setRoi({ ...roi, [k]: parseInt(e.target.value) })}
                                                style={{ width: '100%', padding: '10px', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', borderRadius: '6px', textAlign: 'center', fontSize: '1rem' }}
                                            />
                                        </div>
                                    ))}
                                </div>
                            </div>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                                {/* Mode Selector */}
                                <div style={{ display: 'flex', gap: '10px' }}>
                                    <button
                                        onClick={() => setAugmentationMode('single')}
                                        style={{ flex: 1, padding: '8px', borderRadius: '6px', border: '1px solid transparent', cursor: 'pointer', background: augmentationMode === 'single' ? 'var(--primary)' : 'rgba(255,255,255,0.05)', color: 'white', transition: 'all 0.2s' }}
                                    >
                                        Standard Mode
                                    </button>
                                    <button
                                        onClick={() => setAugmentationMode('composition')}
                                        style={{ flex: 1, padding: '8px', borderRadius: '6px', border: '1px solid transparent', cursor: 'pointer', background: augmentationMode === 'composition' ? 'var(--primary)' : 'rgba(255,255,255,0.05)', color: 'white', transition: 'all 0.2s' }}
                                    >
                                        Composition Mode
                                    </button>
                                </div>

                                {/* Custom Output Name */}
                                <div style={{ marginBottom: '15px' }}>
                                    <div style={{ fontSize: '0.85rem', opacity: 0.8, marginBottom: '5px' }}>Output Folder Name (Optional)</div>
                                    <input
                                        type="text"
                                        placeholder="e.g. augmented_dataset_v1"
                                        value={customOutputName}
                                        onChange={e => setCustomOutputName(e.target.value)}
                                        style={{
                                            width: '100%',
                                            padding: '10px',
                                            background: 'rgba(0,0,0,0.2)',
                                            border: '1px solid rgba(255,255,255,0.1)',
                                            color: 'white',
                                            borderRadius: '6px',
                                            fontSize: '0.9rem'
                                        }}
                                    />
                                    <div style={{ fontSize: '0.75rem', opacity: 0.5, marginTop: '3px' }}>
                                        If left empty, defaults to 'augmented' subfolder.
                                    </div>
                                </div>

                                {augmentationMode === 'single' ? (
                                    <>
                                        <div style={{ display: 'flex', gap: '20px', flexDirection: 'column' }}>
                                            <div style={{ fontSize: '0.9rem' }}>Output Count</div>
                                            <input type="number" min="1" max="100" value={numAugmentations} onChange={e => setNumAugmentations(parseInt(e.target.value))}
                                                style={{ width: '100%', padding: '10px', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', borderRadius: '6px', textAlign: 'center', fontSize: '1rem' }}
                                            />
                                        </div>

                                        {stats && (
                                            <div style={{ marginTop: '10px', padding: '15px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', fontSize: '0.9rem', border: '1px solid rgba(255,255,255,0.1)' }}>
                                                <h5 style={{ margin: '0 0 10px 0', opacity: 0.8 }}>Output Estimate</h5>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                                                    <span style={{ opacity: 0.6 }}>Selected Objects:</span>
                                                    <strong>{selectedClasses.reduce((acc, cid) => acc + (stats.class_counts[cid] || 0), 0).toLocaleString()}</strong>
                                                </div>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '8px' }}>
                                                    <span style={{ opacity: 0.8 }}>Total Augmented Images:</span>
                                                    <strong style={{ color: 'var(--accent)' }}>
                                                        {(selectedClasses.reduce((acc, cid) => acc + (stats.class_counts[cid] || 0), 0) * numAugmentations).toLocaleString()}
                                                    </strong>
                                                </div>
                                            </div>
                                        )}
                                    </>
                                ) : (
                                    <>
                                        <div style={{ display: 'flex', gap: '15px' }}>
                                            <div style={{ flex: 1 }}>
                                                <div style={{ fontSize: '0.8rem', opacity: 0.7, marginBottom: '5px' }}>Total Images to Generate</div>
                                                <input type="number" min="1" max="1000" value={compTotalImages} onChange={e => setCompTotalImages(parseInt(e.target.value))}
                                                    style={{ width: '100%', padding: '10px', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', borderRadius: '6px', textAlign: 'center', fontSize: '1rem' }}
                                                />
                                            </div>
                                            <div style={{ flex: 1 }}>
                                                <div style={{ fontSize: '0.8rem', opacity: 0.7, marginBottom: '5px' }}>Objects per Image</div>
                                                <input type="number" min="1" max="20" value={compObjectsPerImage} onChange={e => setCompObjectsPerImage(parseInt(e.target.value))}
                                                    style={{ width: '100%', padding: '10px', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', borderRadius: '6px', textAlign: 'center', fontSize: '1rem' }}
                                                />
                                            </div>
                                        </div>
                                        <div style={{ fontSize: '0.75rem', opacity: 0.5, marginTop: '5px', fontStyle: 'italic' }}>
                                            * Objects are randomly selected from current class selection. Collision detection ensures no overlap.
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
                {/* 5. Pipeline (Full Width Row) */}
                <div className="glass-panel" style={{ padding: '20px', minHeight: '350px', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '25px' }}>
                        <h3 style={{ margin: 0 }}>5. Augmentation Pipeline</h3>
                        <div className="dropdown" style={{ position: 'relative' }}>
                            <button
                                className="btn-add-step"
                                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                                style={{ width: 'auto', padding: '12px 30px', height: 'auto', fontSize: '1rem' }}
                            >
                                + Add Operation
                            </button>
                            {isDropdownOpen && (
                                <div className="dropdown-content" style={{ display: 'block', right: 0, left: 'auto', width: '220px', top: '100%', marginTop: '5px' }}>
                                    {AUGMENTATION_TYPES.map(type => {
                                        const isAdded = pipeline.some(p => p.type === type.id);
                                        return (
                                            <div
                                                key={type.id}
                                                onClick={() => !isAdded && startAdding(type.id)}
                                                className="dropdown-item"
                                                style={{
                                                    padding: '12px 15px',
                                                    opacity: isAdded ? 0.5 : 1,
                                                    cursor: isAdded ? 'not-allowed' : 'pointer',
                                                    background: isAdded ? 'rgba(0,0,0,0.2)' : undefined
                                                }}
                                                title={isAdded ? "Already added to pipeline" : type.desc}
                                            >
                                                <span style={{ width: '25px', textAlign: 'center', fontSize: '1.2rem' }}>{type.icon}</span>
                                                {type.label}
                                                {isAdded && <span style={{ marginLeft: 'auto', fontSize: '0.8rem' }}>✓</span>}
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    </div>

                    {
                        pipeline.length === 0 ? (
                            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', border: '2px dashed rgba(255,255,255,0.1)', borderRadius: '15px', opacity: 0.6, minHeight: '200px', background: 'rgba(0,0,0,0.1)' }}>
                                <div style={{ fontSize: '2.5rem', marginBottom: '15px' }}>⚡</div>
                                <div style={{ fontSize: '1.2rem', marginBottom: '5px' }}>Pipeline is empty</div>
                                <div style={{ fontSize: '0.9rem' }}>Add steps to define your augmentation workflow</div>
                            </div>
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                                {pipeline.map((step, idx) => (
                                    <div key={step.id} className="pipeline-card" style={{ width: '100%', padding: '20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(255,255,255,0.02)' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                                            <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '1.2rem' }}>
                                                {idx + 1}
                                            </div>
                                            <div>
                                                <strong style={{ fontSize: '1.1rem', display: 'block', marginBottom: '5px' }}>{step.type.toUpperCase()}</strong>
                                                <div style={{ fontSize: '0.9rem', opacity: 0.7, display: 'flex', gap: '15px' }}>
                                                    {Object.entries(step.params).map(([k, v]) => (
                                                        <span key={k} style={{ background: 'rgba(255,255,255,0.05)', padding: '2px 8px', borderRadius: '4px' }}>
                                                            {k.replace('_range', '')}: <span style={{ color: 'white', fontWeight: '500' }}>{v.join(' - ')}</span>
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                        <button onClick={() => removeStep(idx)} style={{ background: 'rgba(255,0,0,0.15)', border: 'none', color: 'rgb(255, 120, 120)', cursor: 'pointer', width: '36px', height: '36px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '18px', transition: 'all 0.2s' }}>✕</button>
                                    </div>
                                ))}
                            </div>
                        )
                    }

                    <div style={{ marginTop: 'auto', paddingTop: '30px', display: 'flex', justifyContent: 'flex-end', borderTop: '1px solid rgba(255,255,255,0.05)', flexDirection: 'column', gap: '10px' }}>
                        {isTaskRunning ? (
                            <div style={{ width: '100%' }}>
                                <div style={{ fontSize: '0.9rem', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                                    <span>Generating Augmentations...</span>
                                    <span>{taskProgress?.current || 0} / {taskProgress?.total || '?'}</span>
                                </div>
                                <ProgressBar progress={taskProgress} type="augmentation" />
                            </div>
                        ) : (
                            <button className="btn-primary-large" onClick={handleGenerate} disabled={pipeline.length === 0} style={{ width: 'auto', padding: '14px 50px', fontSize: '1.1rem', alignSelf: 'flex-end' }}>
                                🚀 Run Augmentation Task
                            </button>
                        )}
                    </div>
                </div >

                <style>{`
                    .glass-panel { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; box-sizing: border-box; }
                    .btn-browse { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: white; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
                    .path-display { background: rgba(0,0,0,0.2); padding: 6px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05); font-size: 0.85rem; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
                    .pipeline-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; width: 180px; position: relative; }
                    .btn-add-step { background: rgba(255,255,255,0.05); border: 1px dashed rgba(255,255,255,0.3); color: white; padding: 15px; border-radius: 10px; cursor: pointer; width: 150px; height: 100%; display: flex; align-items: center; justify-content: center; font-weight: 500; transition: all 0.2s; }
                    .btn-add-step:hover { background: rgba(255,255,255,0.1); border-color: rgba(255,255,255,0.5); }
                    .dropdown:hover .dropdown-content { display: block; }
                    .dropdown-content { display: none; position: absolute; top: 100%; left: 0; background: #1e1e1e; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); z-index: 100; min-width: 200px; padding: 5px; }
                    .dropdown-item { padding: 10px 15px; cursor: pointer; display: flex; gap: 10px; align-items: center; border-radius: 4px; }
                    .dropdown-item:hover { background: var(--primary); }
                    .btn-primary-large { width: 100%; padding: 15px; font-size: 1.1rem; background: var(--primary); color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
                    .btn-primary-large:disabled { opacity: 0.5; cursor: not-allowed; box-shadow: none; }
                `}</style>
            </div >
        );
    }

    // 2. Editor View (Wizard)
    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <h3>Add {editingType.toUpperCase()} Augmentation</h3>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <button onClick={drawSample} className="btn-browse" disabled={isLoadingSample}>
                        🎲 Draw New Sample
                    </button>
                    <button onClick={cancelAdd} className="btn-browse" style={{ border: 'none' }}>Cancel</button>
                    <button onClick={confirmAdd} className="btn-primary-large" style={{ width: 'auto', padding: '8px 20px', fontSize: '1rem' }}>
                        Confirm & Add
                    </button>
                </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', flex: 1, gap: '20px', minHeight: 0 }}>
                {/* Controls Area (TOP) */}
                <div className="glass-panel" style={{ width: '100%', padding: '20px' }}>
                    <h4 style={{ marginTop: 0, marginBottom: '20px' }}>Parameters</h4>
                    <div style={{ width: '100%' }}>
                        <div style={{ paddingRight: '20px' }}> {/* Extra padding to ensure slider handles don't get clipped */}
                            {editingType === 'rotation' && renderRangeSlider('Rotation Angle', 'rotation_range', -180, 180, 1, '°')}
                            {editingType === 'scaling' && renderRangeSlider('Scale Factor', 'scaling_range', 0.1, 3.0, 0.1, 'x')}
                            {editingType === 'blur' && renderRangeSlider('Kernel Size', 'blur_range', 0, 15, 1, 'px')}
                            {editingType === 'contrast' && renderRangeSlider('Alpha', 'contrast_range', 0.5, 3.0, 0.1)}
                        </div>
                    </div>
                </div>

                {/* Preview Area (BOTTOM - Split View) */}
                <div style={{ flex: 1, display: 'flex', gap: '20px', minHeight: '300px' }}>

                    {/* Left: Original */}
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                        <div style={{ marginBottom: '10px', textAlign: 'center', fontWeight: '500' }}>Original</div>
                        <div className="preview-box">
                            {isLoadingSample ? <div className="spinner"></div> :
                                originalSample ? <img src={originalSample} alt="Original" /> :
                                    <div className="placeholder">No Sample</div>}
                        </div>
                    </div>

                    {/* Right: Augmented */}
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                        <div style={{ marginBottom: '10px', textAlign: 'center', fontWeight: '500' }}>Augmented Preview</div>
                        <div className="preview-box" style={{ border: '2px solid var(--primary)' }}>
                            {isPreviewing ? <div className="spinner"></div> :
                                previewImage ? <img src={previewImage} alt="Augmented" /> :
                                    <div className="placeholder"> Adjust settings to see preview</div>}
                        </div>
                    </div>

                </div>
            </div>

            <style>{`
                .glass-panel { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; }
                .preview-box { flex: 1; background: rgba(0,0,0,0.3); border-radius: 12px; border: 2px dashed rgba(255,255,255,0.1); display: flex; align-items: center; justifyContent: center; overflow: hidden; position: relative; }
                .preview-box img { max-width: 100%; max-height: 100%; object-fit: contain; }
                .placeholder { opacity: 0.3; font-size: 1.2rem; }
                 .btn-primary-large { padding: 15px; background: var(--primary); color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
                 .btn-browse { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: white; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; }
                 .spinner { width: 30px; height: 30px; border: 3px solid rgba(255,255,255,0.3); border-radius: 50%; border-top-color: white; animation: spin 1s ease-in-out infinite; }
                 @keyframes spin { to { transform: rotate(360deg); } }
            `}</style>
        </div>
    );

};

export default DataAugmentationTool;
