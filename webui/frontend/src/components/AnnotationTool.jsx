import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const API_BASE = '/api';
const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'];

const AnnotationTool = ({ datasetPath, onPathChange, samModelPath, setSamModelPath, onBrowse, jumpToImageName, onJumpComplete, onSave }) => {
    const [images, setImages] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [loading, setLoading] = useState(false);
    const [classes, setClasses] = useState([]);
    const [selectedClass, setSelectedClass] = useState(0);

    const [annotations, setAnnotations] = useState([]);
    const [currentTool, setCurrentTool] = useState('cursor'); // cursor, box, polygon
    const [isDrawing, setIsDrawing] = useState(false);
    const [tempPoints, setTempPoints] = useState([]); // Canvas coordinates
    const [samPoints, setSamPoints] = useState([]); // List of {x, y, label} (normalized)
    const [samPreview, setSamPreview] = useState(null); // The returned polygon points (normalized)
    const [samLoading, setSamLoading] = useState(false);
    const [samEpsilon, setSamEpsilon] = useState(2.0);

    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const [imageObj, setImageObj] = useState(null);
    const [scale, setScale] = useState(1);
    const [hoveredIndex, setHoveredIndex] = useState(null);
    const [imageSearchQuery, setImageSearchQuery] = useState('');

    // Fetch initial data
    useEffect(() => {
        if (!datasetPath) return;

        const loadInit = async () => {
            setLoading(true);
            try {
                // Get Classes
                const clsRes = await axios.get(`${API_BASE}/dataset/classes?path=${encodeURIComponent(datasetPath)}`);
                setClasses(clsRes.data.classes);

                // Get Images
                const imgRes = await axios.get(`${API_BASE}/labeled/images?path=${encodeURIComponent(datasetPath)}&limit=10000`);
                const imgList = imgRes.data.images || [];
                setImages(imgList);

                // Only reset to 0 if we are NOT jumping
                if (!jumpToImageName) {
                    setCurrentIndex(0);
                }
            } catch (err) {
                console.error("Error init annotation:", err);
            } finally {
                setLoading(false);
            }
        };
        loadInit();
    }, [datasetPath]);

    // Separate effect for jumping, so it works even if datasetPath doesn't change
    useEffect(() => {
        if (jumpToImageName && images.length > 0) {
            const idx = images.findIndex(img => img.name === jumpToImageName);
            if (idx >= 0) {
                setCurrentIndex(idx);
            }
            if (onJumpComplete) onJumpComplete();
        }
    }, [jumpToImageName, images, onJumpComplete]);

    // Load current image and existing annotations
    useEffect(() => {
        if (images.length === 0 || !datasetPath) {
            setImageObj(null);
            setAnnotations([]);
            return;
        }

        const imgName = images[currentIndex].name;

        const loadData = async () => {
            setTempPoints([]);
            setIsDrawing(false);
            setImageObj(null); // Clear previous to avoid confusion

            // Load Image
            const img = new Image();
            img.onload = () => {
                setImageObj(img);
                fitImage(img);
            };

            img.onerror = () => {
                console.error("Failed to load image:", img.src);
                setImageObj(null);
            };

            // Use timestamp to avoid cache issues
            img.src = `${API_BASE}/annotation/image_file?path=${encodeURIComponent(datasetPath)}&image_name=${encodeURIComponent(imgName)}&t=${new Date().getTime()}`;

            // Load Annotations
            try {
                const annRes = await axios.get(`${API_BASE}/annotation/data?path=${encodeURIComponent(datasetPath)}&image_name=${encodeURIComponent(imgName)}`);
                setAnnotations(annRes.data.annotations);
            } catch (err) {
                setAnnotations([]);
            }
        };
        loadData();
    }, [currentIndex, images, datasetPath]);

    // Keyboard Navigation
    useEffect(() => {
        const handleKeyDown = (e) => {
            // Don't navigate if user is typing in an input/select
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

            if (e.key === 'ArrowRight') {
                setCurrentIndex(prev => Math.min(images.length - 1, prev + 1));
            } else if (e.key === 'ArrowLeft') {
                setCurrentIndex(prev => Math.max(0, prev - 1));
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [images.length]);

    const fitImage = (img) => {
        if (!containerRef.current || !img) return;
        const container = containerRef.current;
        const cw = container.clientWidth || 800;
        const ch = container.clientHeight || 600;
        const scaleW = cw / img.width;
        const scaleH = ch / img.height;
        let newScale = Math.min(scaleW, scaleH);
        if (newScale > 1) newScale = 1;
        if (newScale <= 0) newScale = 0.5; // Fallback
        setScale(newScale);
    };

    // Canvas rendering
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !imageObj) return;

        const ctx = canvas.getContext('2d');
        const drawW = imageObj.width * scale;
        const drawH = imageObj.height * scale;

        canvas.width = drawW;
        canvas.height = drawH;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        try {
            ctx.drawImage(imageObj, 0, 0, drawW, drawH);
        } catch (e) {
            console.error("Canvas draw error:", e);
        }

        // Draw existing annotations
        annotations.forEach((ann, idx) => {
            const isHovered = hoveredIndex === idx;
            const color = COLORS[ann.class_id % COLORS.length];
            ctx.strokeStyle = color;
            ctx.lineWidth = isHovered ? 4 : 2;
            ctx.fillStyle = color + (isHovered ? '80' : '40');

            if (ann.type === 'box') {
                const [cx, cy, w, h] = ann.points;
                const x = (cx - w / 2) * canvas.width;
                const y = (cy - h / 2) * canvas.height;
                const rw = w * canvas.width;
                const rh = h * canvas.height;
                ctx.strokeRect(x, y, rw, rh);
                ctx.fillRect(x, y, rw, rh);

                // Label tag
                ctx.fillStyle = color;
                const label = classes[ann.class_id] || ann.class_id;
                ctx.font = isHovered ? 'bold 12px Inter, system-ui' : '12px Inter, system-ui';
                const textW = ctx.measureText(label).width;
                ctx.fillRect(x, y - 18, textW + 6, 18);
                ctx.fillStyle = 'white';
                ctx.fillText(label, x + 3, y - 5);
            } else {
                ctx.beginPath();
                for (let i = 0; i < ann.points.length; i += 2) {
                    const px = ann.points[i] * canvas.width;
                    const py = ann.points[i + 1] * canvas.height;
                    if (i === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                }
                ctx.closePath();
                ctx.stroke();
                ctx.fill();
            }
        });

        // Draw active drawing
        if (tempPoints.length > 0) {
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);

            if (currentTool === 'box') {
                const [start, end] = tempPoints;
                if (start && end) {
                    const x = Math.min(start.x, end.x);
                    const y = Math.min(start.y, end.y);
                    const w = Math.abs(end.x - start.x);
                    const h = Math.abs(end.y - start.y);
                    ctx.strokeRect(x, y, w, h);
                }
            } else if (currentTool === 'polygon') {
                ctx.beginPath();
                tempPoints.forEach((pt, i) => {
                    if (i === 0) ctx.moveTo(pt.x, pt.y);
                    else ctx.lineTo(pt.x, pt.y);
                });
                // Line to cursor is handled by mousemove state if we added it, 
                // but for now just ensure vertices are distinct
                ctx.stroke();

                // Draw vertices
                tempPoints.forEach(pt => {
                    ctx.fillStyle = 'white';
                    ctx.fillRect(pt.x - 2, pt.y - 2, 4, 4);
                });
            }
            ctx.setLineDash([]);
        }

        // Draw SAM Prompts
        samPoints.forEach(pt => {
            const px = pt.x * canvas.width;
            const py = pt.y * canvas.height;
            ctx.fillStyle = pt.label === 1 ? '#10b981' : '#ef4444';
            ctx.beginPath();
            ctx.arc(px, py, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 1;
            ctx.stroke();
        });

        // Draw SAM Preview
        if (samPreview) {
            ctx.strokeStyle = '#6366f1';
            ctx.lineWidth = 3;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            for (let i = 0; i < samPreview.length; i += 2) {
                const px = samPreview[i] * canvas.width;
                const py = samPreview[i + 1] * canvas.height;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.closePath();
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = 'rgba(99, 102, 241, 0.2)';
            ctx.fill();
        }

    }, [imageObj, scale, annotations, tempPoints, currentTool, classes, hoveredIndex, samPoints, samPreview]);

    const getCanvasCoords = (e) => {
        const rect = canvasRef.current.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    };

    const handleMouseDown = (e) => {
        if (currentTool === 'cursor') return;
        const coords = getCanvasCoords(e);
        if (currentTool === 'box') {
            setIsDrawing(true);
            setTempPoints([coords, coords]);
        } else if (currentTool === 'polygon') {
            setIsDrawing(true);
            setTempPoints(prev => [...prev, coords]);
        } else if (currentTool === 'smart') {
            // Left click = 1 (positive), Right click = 0 (negative)
            if (e.button === 2) e.preventDefault();
            const label = e.button === 2 ? 0 : 1;
            const cw = imageObj.width * scale;
            const ch = imageObj.height * scale;
            const newPoint = { x: coords.x / cw, y: coords.y / ch, label };
            const newPoints = [...samPoints, newPoint];
            setSamPoints(newPoints);
            fetchSAM(newPoints);
        }
    };

    const fetchSAM = async (points) => {
        if (!samModelPath || points.length === 0) return;
        setSamLoading(true);
        try {
            const res = await axios.post(`${API_BASE}/annotation/sam_predict`, {
                model_path: samModelPath,
                image_path: datasetPath,
                image_name: images[currentIndex].name,
                points: points.map(p => [p.x, p.y]),
                labels: points.map(p => p.label),
                epsilon: samEpsilon
            });
            setSamPreview(res.data.points);
        } catch (err) {
            console.error("SAM Error:", err);
        } finally {
            setSamLoading(false);
        }
    };

    const applySAM = () => {
        if (!samPreview) return;
        setAnnotations(prev => [...prev, {
            class_id: selectedClass,
            type: 'polygon',
            points: samPreview
        }]);
        setSamPoints([]);
        setSamPreview(null);
    };

    const handleMouseMove = (e) => {
        if (!isDrawing && currentTool !== 'polygon') return;
        const coords = getCanvasCoords(e);
        if (currentTool === 'box' && isDrawing) {
            setTempPoints(prev => [prev[0], coords]);
        }
    };

    const handleMouseUp = () => {
        if (currentTool === 'box' && isDrawing) {
            const [start, end] = tempPoints;
            if (!start || !end) return;

            const w = Math.abs(end.x - start.x);
            const h = Math.abs(end.y - start.y);
            const cw = imageObj.width * scale;
            const ch = imageObj.height * scale;

            if (w < 5 || h < 5) { setIsDrawing(false); setTempPoints([]); return; }

            const cx = (Math.min(start.x, end.x) + w / 2) / cw;
            const cy = (Math.min(start.y, end.y) + h / 2) / ch;
            const nw = w / cw;
            const nh = h / ch;

            setAnnotations(prev => [...prev, {
                class_id: selectedClass,
                type: 'box',
                points: [cx, cy, nw, nh]
            }]);
            setTempPoints([]);
            setIsDrawing(false);
        }
    };

    const finishPolygon = (e) => {
        if (currentTool === 'polygon' && tempPoints.length > 2) {
            e.preventDefault();
            const cw = imageObj.width * scale;
            const ch = imageObj.height * scale;
            const normPoints = [];
            tempPoints.forEach(pt => {
                normPoints.push(pt.x / cw);
                normPoints.push(pt.y / ch);
            });

            setAnnotations(prev => [...prev, {
                class_id: selectedClass,
                type: 'polygon',
                points: normPoints
            }]);
            setTempPoints([]);
            setIsDrawing(false);
        }
    };

    const handleContextMenu = (e) => {
        e.preventDefault();
        if (currentTool === 'polygon') {
            finishPolygon(e);
        }
    };

    const handleDelete = (idx) => {
        setAnnotations(prev => prev.filter((_, i) => i !== idx));
    };

    const handleSave = async () => {
        if (tempPoints.length > 0) {
            if (!window.confirm("You have an unfinished drawing. It will not be included in the save. Proceed anyway?")) {
                return;
            }
        }
        try {
            await axios.post(`${API_BASE}/annotation/save`, {
                dataset_path: datasetPath,
                image_name: images[currentIndex].name,
                annotations: annotations
            });
            if (onSave) onSave(images[currentIndex].name);
            alert('Saved!');
        } catch (err) {
            alert('Error saving labels');
        }
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '15px' }}>
            <div className="glass" style={{ padding: '10px 20px', display: 'flex', alignItems: 'center', gap: '15px', borderRadius: '12px' }}>
                <span style={{ fontWeight: 'bold' }}>Dataset Path:</span>
                <input
                    type="text"
                    className="input"
                    style={{ flex: 1, background: 'rgba(0,0,0,0.3)', color: 'white', border: '1px solid var(--border-color)', padding: '5px 12px', borderRadius: '6px' }}
                    value={datasetPath}
                    onChange={(e) => onPathChange(e.target.value)}
                    placeholder="Enter dataset path..."
                />
                <button className="btn btn-primary" onClick={onBrowse}>Browse...</button>

                {images.length > 0 && (
                    <div style={{ marginLeft: '10px', paddingLeft: '20px', borderLeft: '1px solid rgba(255,255,255,0.1)' }}>
                        <span style={{ opacity: 0.6, fontSize: '0.85rem' }}>Current Image: </span>
                        <code style={{ color: 'var(--accent)' }}>{images[currentIndex].name}</code>
                    </div>
                )}
            </div>

            <div className="glass" style={{ padding: '10px 20px', display: 'flex', alignItems: 'center', gap: '15px', borderRadius: '12px' }}>
                <span style={{ fontWeight: 'bold' }}>SAM Model:</span>
                <input
                    type="text"
                    className="input"
                    style={{ flex: 1, background: 'rgba(0,0,0,0.3)', color: 'white', border: '1px solid var(--border-color)', padding: '5px 12px', borderRadius: '6px' }}
                    value={samModelPath}
                    onChange={(e) => setSamModelPath(e.target.value)}
                    placeholder="Enter SAM model path (.pt)..."
                />
                <button className="btn btn-primary" onClick={() => onBrowse('sam_model', 'file')}>Browse...</button>
                <div style={{ marginLeft: '10px', paddingLeft: '20px', borderLeft: '1px solid rgba(255,255,255,0.1)' }}>
                    <span style={{ opacity: 0.6, fontSize: '0.85rem' }}>Tool: </span>
                    <code style={{ color: 'var(--accent)' }}>{currentTool}</code>
                </div>
            </div>

            {datasetPath ? (
                <div style={{ display: 'flex', flex: 1, gap: '20px', overflow: 'hidden' }}>
                    {/* Toolbar */}
                    <div className="glass" style={{ width: '60px', borderRadius: '12px', padding: '10px', display: 'flex', flexDirection: 'column', gap: '10px', alignItems: 'center' }}>
                        {['cursor', 'box', 'polygon', 'smart'].map(tool => (
                            <button
                                key={tool}
                                className={`btn ${currentTool === tool ? 'btn-primary' : ''}`}
                                onClick={() => {
                                    setCurrentTool(tool);
                                    if (tool !== 'smart') {
                                        setSamPoints([]);
                                        setSamPreview(null);
                                    }
                                }}
                                title={tool}
                            >
                                {tool === 'cursor' ? '👆' : tool === 'box' ? '⬜' : tool === 'polygon' ? '📐' : '✨'}
                            </button>
                        ))}
                    </div>

                    {/* Canvas Area */}
                    <div ref={containerRef} className="glass" style={{ flex: 1, borderRadius: '12px', overflow: 'hidden', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000' }} onContextMenu={handleContextMenu}>
                        {loading ? (
                            <div style={{ color: 'white', opacity: 0.5 }}>Loading dataset...</div>
                        ) : imageObj ? (
                            <>
                                <canvas
                                    ref={canvasRef}
                                    onMouseDown={handleMouseDown}
                                    onMouseMove={handleMouseMove}
                                    onMouseUp={handleMouseUp}
                                    style={{
                                        cursor: currentTool === 'cursor' ? 'default' : 'crosshair',
                                        border: '1px solid rgba(255,255,255,0.2)',
                                        boxShadow: '0 0 20px rgba(0,0,0,0.5)'
                                    }}
                                />
                                <div style={{ position: 'absolute', top: '10px', right: '10px', background: 'rgba(0,0,0,0.7)', padding: '5px 10px', borderRadius: '4px', fontSize: '10px', color: 'rgba(255,255,255,0.7)', pointerEvents: 'none' }}>
                                    {imageObj.width}x{imageObj.height} @ {(scale * 100).toFixed(1)}%
                                </div>
                            </>
                        ) : (
                            <div style={{ color: 'white', opacity: 0.5 }}>
                                {images.length === 0 ? 'No images found' : 'No Image Selected'}
                            </div>
                        )}
                    </div>

                    {/* Sidebar */}
                    <div className="glass" style={{ width: '300px', borderRadius: '12px', padding: '20px', display: 'flex', flexDirection: 'column' }}>
                        <div style={{ marginBottom: '10px', display: 'flex', gap: '10px', alignItems: 'center' }}>
                            <input
                                type="text"
                                className="input"
                                placeholder="Go to image..."
                                value={imageSearchQuery}
                                onChange={(e) => {
                                    setImageSearchQuery(e.target.value);
                                    if (e.target.value) {
                                        const idx = images.findIndex(img => img.name.toLowerCase().includes(e.target.value.toLowerCase()));
                                        if (idx >= 0) setCurrentIndex(idx);
                                    }
                                }}
                                style={{ flex: 1, padding: '5px' }}
                            />
                        </div>
                        <div style={{ marginBottom: '20px', display: 'flex', gap: '10px', alignItems: 'center' }}>
                            <button className="btn" onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))} disabled={currentIndex === 0}>⬅️</button>
                            <span style={{ fontSize: '0.9rem', flex: 1, textAlign: 'center' }}>
                                {currentIndex + 1} / {images.length}
                            </span>
                            <button className="btn" onClick={() => setCurrentIndex(Math.min(images.length - 1, currentIndex + 1))} disabled={currentIndex >= images.length - 1}>➡️</button>
                        </div>

                        <h3 style={{ margin: '0 0 10px 0' }}>Classes</h3>
                        <select style={{ width: '100%', padding: '10px', background: 'rgba(255,255,255,0.1)', color: 'white', border: '1px solid var(--border-color)', borderRadius: '6px' }} value={selectedClass} onChange={(e) => setSelectedClass(parseInt(e.target.value))}>
                            {classes.map((name, i) => <option key={i} value={i}>{i}: {name}</option>)}
                        </select>

                        <h3 style={{ margin: '20px 0 10px 0' }}>Annotations ({annotations.length})</h3>
                        <div style={{ flex: 1, overflowY: 'auto', marginBottom: '20px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', padding: '5px' }}>
                            {annotations.length === 0 ? (
                                <div style={{ padding: '20px', textAlign: 'center', opacity: 0.5, fontSize: '0.8rem' }}>No annotations yet</div>
                            ) : (
                                annotations.map((ann, i) => (
                                    <div
                                        key={i}
                                        style={{
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            padding: '8px',
                                            borderBottom: '1px solid rgba(255,255,255,0.05)',
                                            fontSize: '0.85rem',
                                            alignItems: 'center',
                                            background: hoveredIndex === i ? 'rgba(255,255,255,0.1)' : 'transparent',
                                            transition: 'background 0.2s'
                                        }}
                                        onMouseEnter={() => setHoveredIndex(i)}
                                        onMouseLeave={() => setHoveredIndex(null)}
                                    >
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                            <div style={{ width: '12px', height: '12px', borderRadius: '2px', background: COLORS[ann.class_id % COLORS.length] }}></div>
                                            <span style={{ fontWeight: 500 }}>
                                                {ann.type === 'box' ? 'Box' : 'Poly'}
                                            </span>
                                            <span style={{ opacity: 0.7 }}>
                                                {classes[ann.class_id] || `Class ${ann.class_id}`}
                                            </span>
                                        </div>
                                        <button
                                            style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', padding: '2px 5px' }}
                                            onClick={() => handleDelete(i)}
                                            title="Delete annotation"
                                        >
                                            ✖️
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>

                        {currentTool === 'polygon' && tempPoints.length > 0 && (
                            <div style={{ marginBottom: '10px', padding: '10px', background: 'rgba(99, 102, 241, 0.2)', borderRadius: '6px', fontSize: '0.8rem', border: '1px solid rgba(99, 102, 241, 0.4)' }}>
                                💡 Tip: <strong>Right-click</strong> to finish your polygon ({tempPoints.length} points so far)
                            </div>
                        )}

                        {currentTool === 'smart' && (
                            <div style={{ marginBottom: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                <div style={{ padding: '10px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '6px', fontSize: '0.8rem', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                                    ✨ <strong>Smart Tool</strong>: <br />
                                    • Left-click: Add object <br />
                                    • Right-click: Remove area
                                </div>

                                <div style={{ marginBottom: '10px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '5px' }}>
                                        <span>Simplification</span>
                                        <span style={{ color: 'var(--accent)' }}>{samEpsilon.toFixed(1)}px</span>
                                    </div>
                                    <input
                                        type="range"
                                        min="0"
                                        max="10"
                                        step="0.5"
                                        value={samEpsilon}
                                        onChange={(e) => {
                                            const val = parseFloat(e.target.value);
                                            setSamEpsilon(val);
                                            // Re-trigger prediction if we have points
                                            if (samPoints.length > 0) fetchSAM(samPoints);
                                        }}
                                        style={{ width: '100%' }}
                                    />
                                </div>

                                <div style={{ display: 'flex', gap: '8px' }}>
                                    <button
                                        className="btn"
                                        style={{ flex: 1, fontSize: '0.8rem' }}
                                        onClick={() => { setSamPoints([]); setSamPreview(null); }}
                                    >
                                        🧹 Clear
                                    </button>
                                    <button
                                        className="btn btn-primary"
                                        style={{ flex: 1, fontSize: '0.8rem' }}
                                        onClick={applySAM}
                                        disabled={!samPreview || samLoading}
                                    >
                                        {samLoading ? '...' : '✅ Apply'}
                                    </button>
                                </div>
                            </div>
                        )}

                        <button
                            className="btn btn-primary"
                            onClick={handleSave}
                            style={{ marginBottom: '10px', width: '100%' }}
                            disabled={images.length === 0}
                        >
                            💾 Save Labels
                        </button>
                    </div>
                </div>
            ) : (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.5 }}>
                    <div style={{ textAlign: 'center' }}>
                        <h3>Manual Annotation Tool</h3>
                        <p>Click "Browse" above to select a folder containing YOLO images and labels.</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AnnotationTool;
