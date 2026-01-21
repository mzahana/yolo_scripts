import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import ReactCrop, { centerCrop, makeAspectCrop } from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    PieChart, Pie, Cell, LabelList
} from 'recharts';

import AnnotationTool from './components/AnnotationTool';

import ProjectLanding from './components/ProjectLanding';

const API_BASE = 'http://localhost:8000/api';
const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'];

function App() {
    const [datasetPath, setDatasetPath] = useState('');
    const [annotationPath, setAnnotationPath] = useState('');
    const [samModelPath, setSamModelPath] = useState('');
    const [jumpToImageName, setJumpToImageName] = useState(null);
    const [datasetInfo, setDatasetInfo] = useState(null);
    const [sampleImage, setSampleImage] = useState(null);

    const [crop, setCrop] = useState({ unit: '%', width: 80, height: 80, x: 10, y: 10 });
    const [completedCrop, setCompletedCrop] = useState(null);
    const [imgRef, setImgRef] = useState(null);
    const [scaledDisplay, setScaledDisplay] = useState(null);

    const [resizeWidth, setResizeWidth] = useState(640);
    const [resizeHeight, setResizeHeight] = useState(640);
    const [lastProcessedDir, setLastProcessedDir] = useState('');

    const [modelPath, setModelPath] = useState('');
    const [confidence, setConfidence] = useState(0.5);

    const [isTaskRunning, setIsTaskRunning] = useState(false);
    const [taskProgress, setTaskProgress] = useState(null);
    const [labelResult, setLabelResult] = useState(null);
    const [hasLabels, setHasLabels] = useState(false);
    const [hasMasks, setHasMasks] = useState(false);
    const [notification, setNotification] = useState('');

    const [maskedImages, setMaskedImages] = useState([]);
    const [maskedPath, setMaskedPath] = useState('');
    const [maskedMountUrl, setMaskedMountUrl] = useState('');
    const [maskedOffset, setMaskedOffset] = useState(0);
    const [totalMasked, setTotalMasked] = useState(0);
    const [projectConfig, setProjectConfig] = useState(null);
    const [projectPaths, setProjectPaths] = useState(null);
    const [landingCallback, setLandingCallback] = useState(null);
    const MASKED_LIMIT = 20;

    // Lightbox & Stats State
    const [selectedLightboxImage, setSelectedLightboxImage] = useState(null);
    const [datasetStats, setDatasetStats] = useState(null);
    const [statsLoading, setStatsLoading] = useState(false);

    // Cache buster for images
    const [cacheBuster, setCacheBuster] = useState(Date.now());

    // Navigation State
    const [activeTab, setActiveTab] = useState('project_home');
    const [navItems, setNavItems] = useState([
        { type: 'header', label: 'PROJECT' },
        { id: 'project_home', label: 'Project Info', icon: '🏠' },
        { type: 'header', label: 'ANNOTATION' },
        { id: 'annotation', label: 'Manual Annotation', icon: '✏️' },
        { id: 'labeling', label: 'Auto-Labeling', icon: '🤖' },
        { type: 'header', label: 'PROCESSING' },
        { id: 'preprocess', label: 'Pre-processing', icon: '✂️' },
        { id: 'dataset_gen', label: 'Create Dataset', icon: '📦' },
        { id: 'processing', label: 'Data Tools', icon: '⚙️' },
        { type: 'header', label: 'ANALYSIS' },
        { id: 'verification', label: 'Data Inspection', icon: '✅' },
        { id: 'stats', label: 'Statistics', icon: '📊' },
    ]);
    const [newDatasetName, setNewDatasetName] = useState('dataset_v1');


    // File Browser State
    const [showFileBrowser, setShowFileBrowser] = useState(false);
    const [browserPath, setBrowserPath] = useState('');
    const [browserItems, setBrowserItems] = useState([]);
    const [browserType, setBrowserType] = useState('dir'); // 'dir' or 'file'
    const [browserTarget, setBrowserTarget] = useState(''); // 'dataset' or 'model'
    const [browserLoading, setBrowserLoading] = useState(false);

    // Merge State
    // Data Processing State
    const [mergeSources, setMergeSources] = useState(['']);
    const [mergeOutput, setMergeOutput] = useState('');
    const [browserIndex, setBrowserIndex] = useState(-1); // -1 for output, >=0 for sources
    const [extractSource, setExtractSource] = useState('');
    const [extractOutput, setExtractOutput] = useState('');
    const [availableClasses, setAvailableClasses] = useState([]);
    const [selectedClasses, setSelectedClasses] = useState([]);
    const [splitInput, setSplitInput] = useState('');
    const [splitCount, setSplitCount] = useState(2);

    // Verification Filtering
    const [filterClasses, setFilterClasses] = useState([]);

    // Stats State
    const [statsPath, setStatsPath] = useState('');

    // Existing Datasets

    // Existing Datasets
    const [existingDatasets, setExistingDatasets] = useState([]);

    const fetchDatasets = async () => {
        if (!datasetPath) return; // Only if project path is set
        try {
            const res = await axios.get(`${API_BASE}/dataset/list?project_path=${encodeURIComponent(datasetPath)}`);
            setExistingDatasets(res.data);
        } catch (err) {
            console.error("Failed to list datasets", err);
        }
    };

    // Single Auto-Label State
    const [labelingMode, setLabelingMode] = useState('batch'); // 'batch' or 'single'
    const [labelingImages, setLabelingImages] = useState([]);
    const [currentSingleImageIndex, setCurrentSingleImageIndex] = useState(0);
    const [singleLabelResult, setSingleLabelResult] = useState(null);
    const [singleLabelLoading, setSingleLabelLoading] = useState(false);
    const [labelingSourcePath, setLabelingSourcePath] = useState('');
    const [imageSearchQuery, setImageSearchQuery] = useState('');

    const showNotification = (msg) => {
        setNotification(msg);
        setTimeout(() => setNotification(''), 3000);
    };

    // Polling for progress
    useEffect(() => {
        let interval;
        if (isTaskRunning) {
            interval = setInterval(async () => {
                try {
                    const res = await axios.get(`${API_BASE}/progress`);
                    const data = res.data;
                    setTaskProgress(data);

                    if (data.status === 'idle' || data.status === 'error') {
                        setIsTaskRunning(false);
                        if (data.result) {
                            const msg = data.message || '';
                            if (msg.toLowerCase().includes('pre-processing') || msg.toLowerCase().includes('finished')) {
                                if (data.result.output_dir) {
                                    setLastProcessedDir(data.result.output_dir);
                                }
                            }

                            if (data.result.labeled_dir && data.result.classes) {
                                setLabelResult(data.result);
                                // On auto-label finish, fetch images
                                if (data.result.masked_dir) {
                                    handleLoadMaskedImages(data.result.masked_dir);
                                    setHasMasks(true);
                                }
                            }

                            if (data.result.masked_dir) {
                                setHasMasks(true);
                                setHasLabels(true);
                                if (!labelResult) {
                                    setLabelResult({
                                        masked_dir: data.result.masked_dir,
                                        labeled_dir: data.result.labeled_dir || '',
                                        classes: [],
                                        yaml_path: 'Auto-detected'
                                    });
                                } else {
                                    setLabelResult(prev => ({
                                        ...prev,
                                        masked_dir: data.result.masked_dir,
                                        labeled_dir: data.result.labeled_dir || prev.labeled_dir
                                    }));
                                }
                                handleLoadMaskedImages(data.result.masked_dir);
                            }
                        }
                    }
                } catch (err) {
                    console.error('Progress polling error:', err);
                    setIsTaskRunning(false);
                }
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [isTaskRunning]);

    const handleCreateProject = async (data) => {
        try {
            const res = await axios.post(`${API_BASE}/project/create`, data);
            handleLoadProject(res.data.path);
        } catch (err) {
            console.error(err);
            alert("Error creating project: " + (err.response?.data?.detail || err.message));
        }
    };

    const handleLoadProject = async (path) => {
        try {
            const res = await axios.post(`${API_BASE}/project/load`, { path });
            setProjectConfig(res.data.config);
            setProjectPaths(res.data.paths);
            setActiveTab('project_home');
            setDatasetPath(res.data.path);

            // Set derived state
            if (res.data.paths.processed) {
                setLastProcessedDir(res.data.paths.processed);
            }
        } catch (err) {
            console.error(err);
            alert("Error loading project: " + (err.response?.data?.detail || err.message));
        }
    };

    // Auto-predict annotation path when dataset path changes
    useEffect(() => {
        if (projectConfig && projectPaths) {
            // In Project mode, 'annotation' tab should point to processed images for labeling
            // The AnnotationTool will load images from here and save labels to project/annotations
            if (projectPaths.processed) {
                setAnnotationPath(projectPaths.processed);
                // Also ensure we set a default class list if available (for the tool to pick up? 
                // AnnotationTool picks up classes from data.yaml usually, but strictly speaking 
                // we might want to pass 'classes' prop if supported. 
                // Currently AnnotationTool fetches classes from 'datasetPath'.
                // We rely on 'datasetPath' (the processed folder) having a data.yaml or classes.txt?
                // No, in project mode, classes are in project root. 
                // The backend 'save_annotation' should handle this.
            }
            return;
        }

        if (datasetPath && datasetPath.trim() !== '') {
            // Predict annotation path: <folder>_processed_labeled
            const baseDir = datasetPath.replace(/\/+$/, '');
            const predicted = baseDir + '_processed_labeled';
            setAnnotationPath(predicted);
        }
    }, [datasetPath, projectConfig, projectPaths]);

    // Project Mode: Handle Tab Switches to load necessary data
    useEffect(() => {
        if (!projectConfig || !projectPaths) return;

        if (activeTab === 'preprocess') {
            // Load sample from RAW images for cropping config
            const rawPath = projectPaths.raw;
            if (rawPath) {
                axios.get(`${API_BASE}/dataset/sample?path=${encodeURIComponent(rawPath)}`)
                    .then(res => {
                        setSampleImage(res.data);
                        // Also set crop from config if available
                        if (projectConfig.crop && res.data.width) {
                            const c = projectConfig.crop;
                            setCrop({
                                unit: '%',
                                x: (c.x / res.data.width) * 100,
                                y: (c.y / res.data.height) * 100,
                                width: (c.width / res.data.width) * 100,
                                height: (c.height / res.data.height) * 100
                            });
                        }
                    })
                    .catch(err => console.error("Failed to load sample for preprocess:", err));

                // Set resize dimensions from config immediately
                console.log("DEBUG: Preprocess Config Check:", projectConfig);
                if (projectConfig) {
                    if (projectConfig.resize_width !== undefined) {
                        console.log("DEBUG: Setting resizeWidth to", projectConfig.resize_width);
                        setResizeWidth(projectConfig.resize_width);
                    }
                    if (projectConfig.resize_height !== undefined) {
                        console.log("DEBUG: Setting resizeHeight to", projectConfig.resize_height);
                        setResizeHeight(projectConfig.resize_height);
                    }
                    if (projectConfig.model_path) {
                        setModelPath(projectConfig.model_path);
                    }
                }
            }
        }


        if (activeTab === 'verification') {
            // Data Inspection: Load masked images from project
            if (projectPaths.masked) {
                handleLoadMaskedImages(projectPaths.masked);
            }
        }

        if (activeTab === 'stats') {
            fetchDatasets();
        }
    }, [activeTab, projectConfig, projectPaths]);

    // Update scaledDisplay when crop or sampleImage changes
    useEffect(() => {
        if (sampleImage && sampleImage.width && crop.width && crop.height) {
            const scaleX = sampleImage.width / 100;
            const scaleY = sampleImage.height / 100;

            setScaledDisplay({
                x: Math.round(crop.x * scaleX),
                y: Math.round(crop.y * scaleY),
                width: Math.round(crop.width * scaleX),
                height: Math.round(crop.height * scaleY)
            });
        }
    }, [crop, sampleImage]);


    const handleLoadMaskedImages = async (path, offset = 0, append = false) => {
        try {
            if (offset === 0) {
                const mountRes = await axios.post(`${API_BASE}/mount?name=masked&path=${encodeURIComponent(path)}`);
                setMaskedMountUrl(mountRes.data.url);
                setMaskedPath(path);
                setCacheBuster(Date.now()); // Update cache buster on new path load
            }

            let url = `${API_BASE}/labeled/images?path=${encodeURIComponent(path)}&limit=${MASKED_LIMIT}&offset=${offset}`;
            if (filterClasses.length > 0) {
                url += `&classes=${encodeURIComponent(filterClasses.join(','))}`;
            }

            const imagesRes = await axios.get(url);
            if (append) {
                setMaskedImages(prev => [...prev, ...imagesRes.data.images]);
            } else {
                setMaskedImages(imagesRes.data.images);
            }
            setTotalMasked(imagesRes.data.total);
            setMaskedOffset(offset);
        } catch (err) {
            console.error('Error loading masked images:', err);
        }
    };

    const handleLoadMore = () => {
        const path = maskedPath || labelResult?.masked_dir;
        if (path) {
            handleLoadMaskedImages(path, maskedOffset + MASKED_LIMIT, true);
        }
    };

    const handleFilterImage = async (imgName) => {
        if (!maskedPath) return;
        try {
            await axios.post(`${API_BASE}/labeled/filter`, {
                image_name: imgName,
                source_dir: maskedPath,
                labeled_dir: labelResult?.labeled_dir
            });
            showNotification('Image copied to filtered folder');
        } catch (err) {
            showNotification('Error filtering image: ' + (err.response?.data?.detail || err.message));
        }
    };

    const fetchStats = async (pathOverride = null) => {
        const targetPath = pathOverride || statsPath || labelResult?.labeled_dir;

        if (!targetPath) {
            // Need a path. If project mode, verify logic.
            if (projectPaths?.annotations) {
                // Fallback to annotations if nothing else
                // But wait, user might not have set it.
            } else {
                console.error("fetchStats: targetPath missing");
                return;
            }
        }

        const actualPath = targetPath || projectPaths?.annotations;
        if (!actualPath) return;

        setStatsLoading(true);
        try {
            console.log("Fetching stats for:", actualPath);
            const res = await axios.get(`${API_BASE}/labeled/stats?path=${encodeURIComponent(actualPath)}`);
            setDatasetStats(res.data);
            setStatsPath(actualPath); // Sync state
            setActiveTab('stats');
        } catch (err) {
            console.error("Error fetching stats:", err);
            showNotification('Error fetching stats: ' + (err.response?.data?.detail || err.message));
        } finally {
            setStatsLoading(false);
        }
    };

    const loadDatasetInfo = async () => {
        // Reset state for new dataset
        setDatasetInfo(null);
        setSampleImage(null);
        setLabelResult(null);
        setMaskedImages([]);
        setMaskedPath('');
        setMaskedMountUrl('');
        setDatasetStats(null);
        setTaskProgress(null);
        setIsTaskRunning(false);
        setHasMasks(false); // Reset hasMasks
        setHasLabels(false);
        setProjectConfig(null);

        try {
            const res = await axios.post(`${API_BASE}/dataset/info`, { path: datasetPath });
            setDatasetInfo(res.data);

            await axios.post(`${API_BASE}/mount?name=dataset&path=${encodeURIComponent(datasetPath)}`);
            const sampleRes = await axios.get(`${API_BASE}/dataset/sample?path=${encodeURIComponent(datasetPath)}`);
            setSampleImage(sampleRes.data);
            setCrop({ unit: '%', width: 80, height: 80, x: 10, y: 10 });

            // Check for existing processing/labeling
            const statusRes = await axios.get(`${API_BASE}/dataset/status?path=${encodeURIComponent(datasetPath)}`);
            const { processed_dir, labeled_dir, labels_root, has_labels, has_masks, config } = statusRes.data;

            setHasMasks(has_masks);
            setHasLabels(has_labels);
            setProjectConfig(config);

            // Fetch classes from data.yaml for filtering/extraction
            // Try labeled_dir first as it's the most likely source of truth for the gallery
            // We run this BEFORE any potential early returns from config restoration
            const runClassDiscovery = async (lDir, dPath) => {
                let classSearchPath = lDir || dPath;
                try {
                    const classRes = await axios.get(`${API_BASE}/dataset/classes?path=${encodeURIComponent(classSearchPath)}`);
                    setAvailableClasses(classRes.data.classes);
                } catch (err) {
                    if (lDir) {
                        try {
                            const classRes = await axios.get(`${API_BASE}/dataset/classes?path=${encodeURIComponent(dPath)}`);
                            setAvailableClasses(classRes.data.classes);
                        } catch (err2) {
                            console.error("Failed to fetch classes for dataset info (both paths):", err2);
                        }
                    } else {
                        console.error("Failed to fetch classes for dataset info:", err);
                    }
                }
            };

            await runClassDiscovery(labeled_dir, datasetPath);

            // Prioritize config if available, but only if the files still exist
            if (config) {
                if (config.processed_dir) setLastProcessedDir(config.processed_dir);
                if (config.model_path) setModelPath(config.model_path);

                // NEW: Load crop and resize settings
                if (config.resize_width) setResizeWidth(config.resize_width);
                if (config.resize_height) setResizeHeight(config.resize_height);
                if (config.crop && sampleRes.data.width && sampleRes.data.height) {
                    const c = config.crop;
                    const pc = {
                        unit: '%',
                        x: (c.x / sampleRes.data.width) * 100,
                        y: (c.y / sampleRes.data.height) * 100,
                        width: (c.width / sampleRes.data.width) * 100,
                        height: (c.height / sampleRes.data.height) * 100,
                    };
                    setCrop(pc);
                    setCompletedCrop(pc);
                }

                // If config has masked_dir AND the backend says we have masks, use it
                if (config.masked_dir && has_masks) {
                    setLabelResult({
                        masked_dir: config.masked_dir,
                        labeled_dir: config.labeled_dir || labeled_dir || labels_root,
                        classes: [],
                        yaml_path: 'From Config'
                    });
                    handleLoadMaskedImages(config.masked_dir);
                    showNotification('Restored project state from config file');
                    return;
                }
            }

            if (processed_dir) {
                setLastProcessedDir(processed_dir);
            }

            if (labeled_dir) {
                // Determine if this is a path to actual masks or just the labeled root
                // If has_masks is true, handleLoadMaskedImages will work
                setLabelResult({
                    masked_dir: has_masks ? labeled_dir : null,
                    labeled_dir: labeled_dir, // This is where labels are expected to be nearby
                    classes: [],
                    yaml_path: 'Auto-detected'
                });

                if (has_masks) {
                    handleLoadMaskedImages(labeled_dir);
                    showNotification('Found existing labeled data and auto-loaded verification gallery');
                } else if (has_labels) {
                    showNotification('Found existing labels. You can generate masked images in the Verification tab.');
                }
            } else if (processed_dir) {
                showNotification('Found existing processed dataset');
            } else {
                showNotification('Dataset loaded successfully');
            }

        } catch (err) {
            showNotification('Error loading dataset: ' + (err.response?.data?.detail || err.message));
        }
    };

    const jumpToAnnotation = (imageName) => {
        // If annotation path is not set, use main dataset path
        if (!annotationPath && datasetPath) {
            setAnnotationPath(datasetPath);
        }
        setJumpToImageName(imageName);
        setActiveTab('annotation');
    };

    const handleLandingBrowse = (cb, type) => {
        setLandingCallback(() => cb);
        openFileBrowser('landing_generic', type);
    };

    // File Browser Logic
    const openFileBrowser = (target, type, index = -1) => {
        setBrowserTarget(target);
        setBrowserType(type);
        setBrowserIndex(index);

        let initialPath = '/';
        if (target === 'dataset') initialPath = datasetPath || '/';
        else if (target === 'model') initialPath = modelPath ? modelPath.substring(0, modelPath.lastIndexOf('/')) : '/';
        else if (target === 'merge_source' && index >= 0) initialPath = mergeSources[index] || '/';
        else if (target === 'merge_output') initialPath = mergeOutput || '/';
        else if (target === 'extract_source') initialPath = extractSource || '/';
        else if (target === 'extract_output') initialPath = extractOutput || '/';
        else if (target === 'annotation_dataset') initialPath = annotationPath || '/';
        else if (target === 'sam_model') initialPath = samModelPath ? samModelPath.substring(0, samModelPath.lastIndexOf('/')) : '/';

        fetchFS(initialPath);
        setShowFileBrowser(true);
    };

    const fetchFS = async (path) => {
        setBrowserLoading(true);
        try {
            const res = await axios.get(`${API_BASE}/fs/list?path=${encodeURIComponent(path)}&only_dirs=${browserType === 'dir'}`);
            setBrowserItems(res.data.items);
            setBrowserPath(res.data.current_path);
        } catch (err) {
            console.error('FS list error:', err);
            showNotification('Error listing filesystem');
        } finally {
            setBrowserLoading(false);
        }
    };

    const selectPath = (item) => {
        if (item.is_dir) {
            fetchFS(item.path);
        } else if (browserType === 'file') {
            if (browserTarget === 'model') setModelPath(item.path);
            else if (browserTarget === 'sam_model') setSamModelPath(item.path);
            setShowFileBrowser(false);
        }
    };

    const handleConfirmFolder = () => {
        if (browserType === 'dir') {
            if (browserTarget === 'dataset') setDatasetPath(browserPath);
            else if (browserTarget === 'annotation_dataset') setAnnotationPath(browserPath);
            else if (browserTarget === 'merge_source' && browserIndex >= 0) {
                const newSources = [...mergeSources];
                newSources[browserIndex] = browserPath;
                setMergeSources(newSources);
            }
            else if (browserTarget === 'merge_output') setMergeOutput(browserPath);
            else if (browserTarget === 'extract_source') setExtractSource(browserPath);
            else if (browserTarget === 'extract_output') setExtractOutput(browserPath);
            else if (browserTarget === 'extract_source') setExtractSource(browserPath);
            else if (browserTarget === 'extract_output') setExtractOutput(browserPath);
            else if (browserTarget === 'extract_output') setExtractOutput(browserPath);
            else if (browserTarget === 'split_input') setSplitInput(browserPath);
            else if (browserTarget === 'landing_generic') {
                if (landingCallback) landingCallback(browserPath);
            }
        }
        setShowFileBrowser(false);
    };

    const FileBrowserModal = () => {
        if (!showFileBrowser) return null;
        return (
            <div className="modal-overlay" onClick={() => setShowFileBrowser(false)}>
                <div className="glass modal-content" onClick={e => e.stopPropagation()}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                        <h3 style={{ margin: 0 }}>
                            {browserTarget === 'dataset' && 'Select Dataset Folder'}
                            {browserTarget === 'model' && 'Select YOLO Model File'}
                            {browserTarget === 'sam_model' && 'Select SAM Model File (.pt)'}
                            {browserTarget === 'merge_source' && 'Select Source Dataset'}
                            {browserTarget === 'merge_output' && 'Select Output Directory'}
                            {browserTarget === 'extract_source' && 'Select Labeled Dataset'}
                            {browserTarget === 'extract_output' && 'Select Extraction Output'}
                            {browserTarget === 'landing_generic' && 'Select Folder'}

                        </h3>
                        <button className="browse-btn" onClick={() => setShowFileBrowser(false)}>Close</button>
                    </div>

                    <div className="path-nav">
                        {browserPath.split('/').map((part, i, arr) => (
                            <span key={i} onClick={() => fetchFS(arr.slice(0, i + 1).join('/') || '/')}>
                                {part || '/'}{i < arr.length - 1 ? ' / ' : ''}
                            </span>
                        ))}
                    </div>

                    <div className="fs-list">
                        {browserLoading ? (
                            <div style={{ padding: '40px', textAlign: 'center' }}>Loading...</div>
                        ) : (
                            browserItems.map((item, i) => (
                                <div key={i} className={`fs-item ${item.is_dir ? '' : 'file-item'}`} onClick={() => selectPath(item)}>
                                    <span className="fs-icon">{item.is_dir ? '📁' : '📄'}</span>
                                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                                        <span>{item.name}</span>
                                        <span style={{ fontSize: '0.7rem', opacity: 0.5 }}>{item.path}</span>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>

                    <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'flex-end', gap: '15px' }}>
                        {browserType === 'dir' && (
                            <button className="btn btn-primary" onClick={handleConfirmFolder}>
                                Select Current {browserTarget.includes('output') ? 'Output' : 'Folder'}
                            </button>
                        )}
                    </div>
                </div>
            </div>
        );
    };

    const calculateScaledCrop = (cropData) => {
        if (!cropData || !sampleImage) return null;
        const { x, y, width, height, unit } = cropData;

        if (unit === '%') {
            return {
                x: Math.round((x / 100) * sampleImage.width),
                y: Math.round((y / 100) * sampleImage.height),
                width: Math.round((width / 100) * sampleImage.width),
                height: Math.round((height / 100) * sampleImage.height),
            };
        } else {
            if (!imgRef) return null;
            const scaleX = sampleImage.width / imgRef.width;
            const scaleY = sampleImage.height / imgRef.height;
            return {
                x: Math.round(x * scaleX),
                y: Math.round(y * scaleY),
                width: Math.round(width * scaleX),
                height: Math.round(height * scaleY),
            };
        }
    };

    const updateManualCrop = (field, value) => {
        if (!sampleImage) return;
        const val = parseInt(value) || 0;
        const newScaled = {
            x: field === 'x' ? val : (scaledDisplay?.x || 0),
            y: field === 'y' ? val : (scaledDisplay?.y || 0),
            width: field === 'width' ? val : (scaledDisplay?.width || 0),
            height: field === 'height' ? val : (scaledDisplay?.height || 0),
        };

        setCrop({
            unit: '%',
            x: (newScaled.x / sampleImage.width) * 100,
            y: (newScaled.y / sampleImage.height) * 100,
            width: (newScaled.width / sampleImage.width) * 100,
            height: (newScaled.height / sampleImage.height) * 100,
        });
        setCompletedCrop({
            unit: '%',
            x: (newScaled.x / sampleImage.width) * 100,
            y: (newScaled.y / sampleImage.height) * 100,
            width: (newScaled.width / sampleImage.width) * 100,
            height: (newScaled.height / sampleImage.height) * 100,
        });
    };

    const handlePreprocess = async () => {
        setIsTaskRunning(true);
        setLabelResult(null);
        setTaskProgress({ status: 'processing', message: 'Starting...', current: 0, total: 100 });
        try {
            // Determine Input/Output based on Project Mode
            let inputPath = datasetPath;
            let outputPath = null;

            if (projectConfig && projectPaths) {
                inputPath = projectPaths.raw;
                outputPath = projectPaths.processed;
            }

            const payload = {
                dataset_path: inputPath,
                output_dir: outputPath,
                resize_width: parseInt(resizeWidth),
                resize_height: parseInt(resizeHeight)
            };
            const scaled = calculateScaledCrop(completedCrop);
            if (scaled) payload.crop = scaled;

            await axios.post(`${API_BASE}/preprocess`, payload);
            showNotification('Pre-processing started...');
        } catch (err) {
            showNotification('Pre-processing failed: ' + (err.response?.data?.detail || err.message));
            setIsTaskRunning(false);
            setTaskProgress(null);
        }
    };

    const handleAutoLabel = async () => {
        if (!modelPath) {
            showNotification('Please select or enter a YOLO model path (.pt)');
            return;
        }
        setIsTaskRunning(true);
        setTaskProgress({ status: 'labeling', message: 'Starting...', current: 0, total: 100 });
        try {
            // Determine Input Path
            let targetPath = lastProcessedDir || datasetPath;
            if (projectPaths && projectPaths.processed) {
                targetPath = projectPaths.processed;
            }

            await axios.post(`${API_BASE}/autolabel`, {
                dataset_path: targetPath,
                model_path: modelPath,
                confidence: parseFloat(confidence),
                save_masked: true
            });
            showNotification('Auto-labeling started...');
        } catch (err) {
            const errorMsg = err.response?.data?.detail || err.message;
            showNotification('Auto-labeling failed: ' + errorMsg);
            setIsTaskRunning(false);
            setTaskProgress(null);
        }
    };

    const handleMerge = async () => {
        const validSources = mergeSources.filter(s => s.trim() !== '');
        if (validSources.length < 2) {
            showNotification('Please select at least two datasets to merge');
            return;
        }
        if (!mergeOutput) {
            showNotification('Please select an output directory');
            return;
        }

        setIsTaskRunning(true);
        setTaskProgress({ status: 'merging', message: 'Starting merge...', current: 0, total: 100 });
        try {
            await axios.post(`${API_BASE}/merge_datasets`, {
                source_paths: validSources,
                output_path: mergeOutput
            });
            showNotification('Merge process started...');
        } catch (err) {
            showNotification('Merge failed: ' + (err.response?.data?.detail || err.message));
            setIsTaskRunning(false);
            setTaskProgress(null);
        }
    };

    const addMergeSource = () => setMergeSources([...mergeSources, '']);
    const removeMergeSource = (index) => {
        const newSources = mergeSources.filter((_, i) => i !== index);
        setMergeSources(newSources.length ? newSources : ['']);
    };


    const handleSplit = async () => {
        try {
            if (!splitInput) {
                setNotification('Please select a dataset to split');
                setTimeout(() => setNotification(''), 3000);
                return;
            }
            if (splitCount < 2) {
                setNotification('Number of splits must be at least 2');
                setTimeout(() => setNotification(''), 3000);
                return;
            }

            setIsTaskRunning(true);
            setTaskProgress({ status: 'splitting', message: 'Initializing split...', current: 0, total: 0 });

            await axios.post(`${API_BASE}/split-dataset`, {
                input_path: splitInput,
                num_splits: parseInt(splitCount)
            });
            // Polling will handle the rest via isTaskRunning
        } catch (err) {
            console.error(err);
            setIsTaskRunning(false);
            setNotification('Error starting split');
            setTimeout(() => setNotification(''), 3000);
        }
    };

    const handleFetchClasses = async () => {
        if (!extractSource) {
            showNotification('Please select a source dataset first');
            return;
        }
        try {
            const res = await axios.get(`${API_BASE}/dataset/classes?path=${encodeURIComponent(extractSource)}`);
            setAvailableClasses(res.data.classes);
            setSelectedClasses([]);
            showNotification(`Found ${res.data.classes.length} classes`);
        } catch (err) {
            showNotification('Failed to fetch classes: ' + (err.response?.data?.detail || err.message));
        }
    };

    const handleExtract = async () => {
        if (!extractSource || !extractOutput || selectedClasses.length === 0) {
            showNotification('Please provide source, output, and select at least one class');
            return;
        }

        setIsTaskRunning(true);
        setTaskProgress({ status: 'extracting', message: 'Starting extraction...', current: 0, total: 100 });
        try {
            await axios.post(`${API_BASE}/extract_by_class`, {
                source_path: extractSource,
                selected_classes: selectedClasses,
                output_path: extractOutput
            });
            showNotification('Extraction process started...');
        } catch (err) {
            showNotification('Extraction failed: ' + (err.response?.data?.detail || err.message));
            setIsTaskRunning(false);
            setTaskProgress(null);
        }
    };

    const handleExtractEmpty = async () => {
        // Validation: If in project mode (StatsView), we use statsPath.
        // If legacy mode, we use datasetPath.
        const sourcePath = activeTab === 'stats' ? statsPath : datasetPath;

        if (!sourcePath) {
            showNotification('Please load a dataset or select a stats source first.');
            return;
        }

        setIsTaskRunning(true);
        setTaskProgress({ status: 'extracting_empty', message: 'Starting extraction of empty images...', current: 0, total: 100 });

        try {
            await axios.post(`${API_BASE}/dataset/extract_empty`, {
                dataset_path: sourcePath, // This will be treated as root/source for path resolution
                labeled_root: activeTab === 'stats' ? (statsPath || projectPaths?.annotations) : labelResult?.labeled_dir
            });
            showNotification('Extraction task started...');
        } catch (err) {
            showNotification('Failed to start extraction: ' + (err.response?.data?.detail || err.message));
            setIsTaskRunning(false);
            setTaskProgress(null);
        }
    };

    const handleGenerateMasks = async () => {
        if (!datasetPath) {
            showNotification('Please load a dataset first.');
            return;
        }
        if (!labelResult?.labeled_dir) {
            showNotification('No labeled directory found. Please run auto-labeling first.');
            return;
        }
        setIsTaskRunning(true);
        setTaskProgress({ status: 'generating_masks', message: 'Starting mask generation...', current: 0, total: 100 });
        try {
            const res = await axios.post(`${API_BASE}/generate_masked`, {
                dataset_path: datasetPath,
                labels_path: labelResult.labeled_dir
            });
            showNotification('Mask generation started...');
        } catch (err) {
            showNotification('Failed to start mask generation: ' + (err.response?.data?.detail || err.message));
            setIsTaskRunning(false);
            setTaskProgress(null);
        }
    };

    const handleRefreshGallery = () => {
        if (maskedPath) {
            setCacheBuster(Date.now());
            handleLoadMaskedImages(maskedPath);
            showNotification('Refreshed gallery metadata');
        }
    };

    const navigateSingleImage = (direction) => {
        if (labelingImages.length === 0) return;

        let newIndex = currentSingleImageIndex + direction;
        // Clamp logic
        if (newIndex < 0) newIndex = 0;
        if (newIndex >= labelingImages.length) newIndex = labelingImages.length - 1;

        if (newIndex !== currentSingleImageIndex) {
            setCurrentSingleImageIndex(newIndex);
            setSingleLabelResult(null); // Reset result
        }
    };

    const handleImageSearch = (e) => {
        const query = e.target.value;
        setImageSearchQuery(query);

        if (!query) return;

        // Find first match
        const index = labelingImages.findIndex(img => img.name.toLowerCase().includes(query.toLowerCase()));
        if (index !== -1) {
            setCurrentSingleImageIndex(index);
            setSingleLabelResult(null);
        }
    };

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (activeTab === 'labeling' && labelingMode === 'single') {
                // Ignore if typing in an input
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                if (e.key === 'ArrowLeft') {
                    navigateSingleImage(-1);
                } else if (e.key === 'ArrowRight') {
                    navigateSingleImage(1);
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [activeTab, labelingMode, labelingImages, currentSingleImageIndex]);

    const fetchLabelingImages = async () => {
        let path = lastProcessedDir;

        // If no explicit last processed dir, try to predict it
        if (!path && datasetPath) {
            // Check for _processed sibling
            const potential = datasetPath.replace(/\/+$/, '') + '_processed';
            try {
                // Quick check if it exists by listing it. 
                // We use only_dirs=true to be lightweight, if it returns items (or empty list) it exists.
                await axios.get(`${API_BASE}/fs/list?path=${encodeURIComponent(potential)}&only_dirs=true`);
                path = potential;
            } catch (ignore) {
                // If it doesn't exist, fall back to datasetPath
                path = datasetPath;
            }
        }

        if (!path) path = datasetPath;
        if (!path) return;

        try {
            const res = await axios.get(`${API_BASE}/fs/list?path=${encodeURIComponent(path)}&only_dirs=false`);
            // Filter for images. Note: fs/list returns names.
            const imgs = res.data.items.filter(i => !i.is_dir && /\.(jpg|jpeg|png|tif|tiff)$/i.test(i.name));
            setLabelingImages(imgs);
            setLabelingSourcePath(path);

            // Mount this path so we can serve images
            await axios.post(`${API_BASE}/mount?name=labeling_source&path=${encodeURIComponent(path)}`);

            if (imgs.length > 0 && currentSingleImageIndex >= imgs.length) {
                setCurrentSingleImageIndex(0);
            }
        } catch (err) {
            console.error("Error fetching labeling images:", err);
            showNotification('Error loading images for labeling');
        }
    };

    // Call this when ensuring we have images
    useEffect(() => {
        if (activeTab === 'labeling' && labelingMode === 'single' && labelingImages.length === 0) {
            fetchLabelingImages();
        }
    }, [activeTab, labelingMode, lastProcessedDir, datasetPath]);

    const handleAutoLabelSingle = async () => {
        if (!modelPath) {
            showNotification('Please select a model first');
            return;
        }
        const img = labelingImages[currentSingleImageIndex];
        if (!img) return;
        setSingleLabelLoading(true);
        setSingleLabelResult(null);

        try {
            // Use the path we determined in fetchLabelingImages
            const sourcePath = labelingSourcePath || lastProcessedDir || datasetPath;

            const res = await axios.post(`${API_BASE}/autolabel/single`, {
                dataset_path: sourcePath,
                image_name: img.name,
                model_path: modelPath,
                confidence: parseFloat(confidence),
                save_masked: true
            });
            setSingleLabelResult(res.data);
            setCacheBuster(Date.now());
            showNotification('Image labeled successfully');
        } catch (err) {
            showNotification('Labeling failed: ' + (err.response?.data?.detail || err.message));
        } finally {
            setSingleLabelLoading(false);
        }
    };

    const toggleClass = (cls) => {
        if (selectedClasses.includes(cls)) {
            setSelectedClasses(selectedClasses.filter(c => c !== cls));
        } else {
            setSelectedClasses([...selectedClasses, cls]);
        }
    };

    const toggleFilterClass = (cls) => {
        setFilterClasses(prev => {
            const next = prev.includes(cls) ? prev.filter(c => c !== cls) : [...prev, cls];
            return next;
        });
    };

    // Reload gallery when filter changes
    useEffect(() => {
        if (maskedPath) {
            handleLoadMaskedImages(maskedPath);
        }
    }, [filterClasses]);

    const ProgressBar = ({ progress, type }) => {
        if (!progress) return null;
        // Show if status matches type OR if it's an error and we were doing this type
        // (Note: we don't have a direct 'previousType' but we can check if progress.status is error)
        if (progress.status !== type && progress.status !== 'error') return null;

        const isError = progress.status === 'error';
        const percentage = progress.total > 0 ? (progress.current / progress.total) * 100 : 0;

        return (
            <div style={{ marginTop: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.8rem', color: isError ? '#ef4444' : 'inherit' }}>
                    <span style={{ fontWeight: isError ? 'bold' : 'normal' }}>
                        {isError ? '❌ ' : ''}{progress.message}
                    </span>
                    {!isError && <span>{Math.round(percentage || 0)}% ({progress.current}/{progress.total})</span>}
                </div>
                {!isError && (
                    <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{
                            width: `${percentage || 0}%`, height: '100%', background: 'var(--primary)',
                            transition: 'width 0.3s ease'
                        }} />
                    </div>
                )}
            </div>
        );
    };

    const VerificationGallery = ({ onJumpToAnnotation }) => {
        // Use labels classes if available
        const classes = availableClasses.length > 0 ? availableClasses : (datasetStats?.class_stats?.map(s => s.name) || []);

        return (
            <div>
                {/* Filter UI */}
                <div className="glass" style={{ padding: '20px', marginBottom: '30px', borderRadius: '12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '15px' }}>
                        <div style={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <span>🔍 Filter by Class</span>
                            {filterClasses.length > 0 && (
                                <button
                                    className="btn"
                                    style={{ padding: '2px 10px', fontSize: '0.7rem', background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444' }}
                                    onClick={() => setFilterClasses([])}
                                >
                                    Clear All
                                </button>
                            )}
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                            <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>
                                Showing {maskedImages.length} of {totalMasked} images
                            </div>
                            <button
                                className="btn btn-secondary"
                                style={{
                                    padding: '4px 12px',
                                    fontSize: '0.75rem',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '6px',
                                    borderColor: 'rgba(255,255,255,0.2)'
                                }}
                                onClick={handleGenerateMasks}
                                disabled={isTaskRunning}
                                title="Force re-generation of all masked images from current labels"
                            >
                                🔄 Re-generate Masks
                            </button>
                            <button
                                className="btn btn-secondary"
                                style={{
                                    padding: '4px 12px',
                                    fontSize: '0.75rem',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '6px',
                                    borderColor: 'rgba(255,255,255,0.2)'
                                }}
                                onClick={handleRefreshGallery}
                                title="Refresh gallery to show latest images and labels"
                            >
                                🔄 Refresh
                            </button>
                        </div>
                    </div>

                    <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                        {classes.map(cls => (
                            <button
                                key={cls}
                                className={`badge ${filterClasses.includes(cls) ? 'active' : ''}`}
                                style={{
                                    cursor: 'pointer',
                                    border: '1px solid var(--border-color)',
                                    background: filterClasses.includes(cls) ? 'var(--primary)' : 'transparent',
                                    color: filterClasses.includes(cls) ? 'white' : 'inherit',
                                    padding: '5px 12px'
                                }}
                                onClick={() => toggleFilterClass(cls)}
                            >
                                {cls}
                            </button>
                        ))}
                        {classes.length === 0 && (
                            <div style={{ fontSize: '0.8rem', opacity: 0.5, fontStyle: 'italic' }}>
                                No classes detected. Run labeling or stats task to see class filters.
                            </div>
                        )}
                    </div>
                </div>

                {/* Content States */}
                {maskedImages.length === 0 && !isTaskRunning && (
                    <div style={{ padding: '20px 0' }}>
                        {hasLabels ? (
                            <div className="glass" style={{ padding: '60px', textAlign: 'center' }}>
                                <h3>{totalMasked === 0 && hasMasks ? "Masked Folder Empty" : "No Masked Images Found"}</h3>
                                <p style={{ opacity: 0.7, marginBottom: '20px' }}>
                                    We found labels but no visual overlays (masked images). To verify the annotations visually, you need to generate images with the labels drawn on them.
                                </p>
                                <button
                                    className="btn btn-primary"
                                    onClick={handleGenerateMasks}
                                    disabled={isTaskRunning}
                                >
                                    📦 Generate Masked Images
                                </button>
                                <ProgressBar progress={taskProgress} type="generating_masks" />
                            </div>
                        ) : (
                            <div style={{ padding: '60px', textAlign: 'center', opacity: 0.5 }}>
                                {!hasMasks && !hasLabels
                                    ? "No labeled images or masks found. Run auto-labeling first."
                                    : "No images match the selected filter."}
                            </div>
                        )}
                    </div>
                )}

                {maskedImages.length > 0 && (
                    <div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '20px' }}>
                            {maskedImages.map((img, i) => (
                                <div key={i} className="glass" style={{ padding: '10px', borderRadius: '12px', display: 'flex', flexDirection: 'column' }}>
                                    <div
                                        style={{ cursor: 'pointer', overflow: 'hidden', borderRadius: '8px' }}
                                        onClick={() => setSelectedLightboxImage(img)}
                                    >
                                        <img
                                            src={`http://localhost:8000${maskedMountUrl}/${img.name}?t=${cacheBuster}`}
                                            style={{ width: '100%', height: 'auto', display: 'block', transition: 'transform 0.3s' }}
                                            className="gallery-img"
                                            alt={img.name}
                                        />
                                    </div>
                                    <div style={{ padding: '10px 0', flex: 1 }}>
                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px', wordBreak: 'break-all', fontFamily: 'monospace' }}>
                                            {img.name}
                                        </div>
                                        <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap', marginBottom: '10px' }}>
                                            {Object.entries(img.stats).map(([cls, count]) => (
                                                <span key={cls} className="badge" style={{ fontSize: '0.65rem', padding: '2px 6px' }}>
                                                    {cls}: {count}
                                                </span>
                                            ))}
                                            {Object.keys(img.stats).length === 0 && (
                                                <span className="badge" style={{ fontSize: '0.65rem', background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444' }}>Empty</span>
                                            )}
                                        </div>
                                        <div style={{ display: 'flex', gap: '8px' }}>
                                            <button
                                                className="btn"
                                                style={{ flex: 1, padding: '8px 5px', fontSize: '0.75rem', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' }}
                                                onClick={() => onJumpToAnnotation(img.name)}
                                                title="Jump to Annotation Tool"
                                            >
                                                ✏️ Edit
                                            </button>
                                            <button
                                                className="btn"
                                                style={{ flex: 1, padding: '8px 5px', fontSize: '0.75rem', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' }}
                                                onClick={() => handleFilterImage(img.name)}
                                                title="Copy to Filtered Folder"
                                            >
                                                📂 Filter
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                        {
                            maskedImages.length < totalMasked && (
                                <div style={{ marginTop: '40px', textAlign: 'center' }}>
                                    <button className="btn btn-primary" onClick={handleLoadMore}>
                                        Load More ({totalMasked - maskedImages.length} remaining)
                                    </button>
                                </div>
                            )
                        }
                    </div>
                )}
            </div>
        );
    };

    // const scaledDisplay = calculateScaledCrop(completedCrop);



    const LightboxModal = () => {
        if (!selectedLightboxImage) return null;
        return (
            <div className="modal-overlay" onClick={() => setSelectedLightboxImage(null)} style={{ background: 'rgba(0,0,0,0.9)', zIndex: 2000 }}>
                <div className="lightbox-content" onClick={e => e.stopPropagation()} style={{ position: 'relative', maxWidth: '90vw', maxHeight: '90vh' }}>
                    <img
                        src={`http://localhost:8000${maskedMountUrl}/${selectedLightboxImage.name}?t=${cacheBuster}`}
                        style={{ width: '100%', height: 'auto', borderRadius: '12px', boxShadow: '0 0 40px rgba(0,0,0,0.5)' }}
                        alt="Enlarged"
                    />
                    <button
                        className="browse-btn"
                        style={{ position: 'absolute', top: '20px', right: '20px', background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(255,255,255,0.2)' }}
                        onClick={() => setSelectedLightboxImage(null)}
                    >
                        ✕ Close
                    </button>
                    <div style={{ padding: '20px 0', color: 'white' }}>
                        <h3 style={{ margin: 0 }}>{selectedLightboxImage.name}</h3>
                        <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                            {Object.entries(selectedLightboxImage.stats).map(([cls, count]) => (
                                <span key={cls} className="badge" style={{ fontSize: '0.9rem' }}>{cls}: {count}</span>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        );
    };


    const StatsView = () => {
        // Prepare options
        const options = [
            { label: 'Entire Project (Current State)', value: projectPaths?.annotations || '' },
            ...existingDatasets.map(ds => ({ label: `Dataset: ${ds.name}`, value: ds.path }))
        ];

        // Handle selection
        const handleSourceChange = (e) => {
            const val = e.target.value;
            if (val) {
                setStatsPath(val);
                fetchStats(val);
            }
        };

        return (
            <div className="stats-container">
                <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <label style={{ fontWeight: 'bold' }}>Stats Source:</label>
                    <select
                        className="input"
                        style={{ maxWidth: '300px' }}
                        value={statsPath || (projectPaths?.annotations || '')}
                        onChange={handleSourceChange}
                    >
                        {options.map((opt, i) => (
                            <option key={i} value={opt.value}>{opt.label}</option>
                        ))}
                    </select>
                    <button className="btn btn-secondary" onClick={() => fetchStats(statsPath)}>🔄 Refresh</button>
                </div>

                {!datasetStats ? (
                    <div style={{ padding: '40px', textAlign: 'center', opacity: 0.5 }}>
                        Select a source to view statistics.
                    </div>
                ) : (
                    <>
                        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', marginBottom: '30px' }}>
                            <div className="stats-card glass">
                                <div style={{ opacity: 0.6, fontSize: '0.9rem' }}>Total Images</div>
                                <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{datasetStats.total_images}</div>
                            </div>
                            <div className="stats-card glass">
                                <div style={{ opacity: 0.6, fontSize: '0.9rem' }}>Total Objects</div>
                                <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{datasetStats.total_objects}</div>
                            </div>
                            <div className="stats-card glass">
                                <div style={{ opacity: 0.6, fontSize: '0.9rem' }}>Empty Images</div>
                                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: datasetStats.empty_count > 0 ? '#ef4444' : 'var(--accent)' }}>
                                    {datasetStats.empty_count}
                                </div>
                            </div>
                        </div>

                        <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '30px' }}>
                            <div className="glass section-card">
                                <div className="section-title" style={{ fontSize: '1.2rem', marginBottom: '20px' }}>Objects per Class</div>
                                <div style={{ height: '350px' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={datasetStats.class_stats} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                            <XAxis dataKey="name" stroke="var(--text-muted)" angle={-45} textAnchor="end" height={80} interval={0} />
                                            <YAxis stroke="var(--text-muted)" />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid var(--border-color)', borderRadius: '8px' }}
                                                itemStyle={{ color: 'white' }}
                                            />
                                            <Bar dataKey="count" fill="var(--primary)" radius={[4, 4, 0, 0]}>
                                                {datasetStats.class_stats.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            <div className="glass section-card">
                                <div className="section-title" style={{ fontSize: '1.2rem', marginBottom: '20px' }}>Class Distribution (%)</div>
                                <div style={{ height: '350px' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <PieChart>
                                            <Pie
                                                data={datasetStats.class_stats}
                                                dataKey="count"
                                                nameKey="name"
                                                cx="50%" cy="50%"
                                                innerRadius={80}
                                                outerRadius={120}
                                                paddingAngle={5}
                                            >
                                                {datasetStats.class_stats.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                                ))}
                                            </Pie>
                                            <Tooltip
                                                contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid var(--border-color)', borderRadius: '8px' }}
                                                itemStyle={{ color: 'white' }}
                                            />
                                            <Legend />
                                        </PieChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>

                        <div className="glass section-card">
                            <div className="section-title" style={{ fontSize: '1.2rem', marginBottom: '20px' }}>Detailed Report</div>
                            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                <thead>
                                    <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border-color)' }}>
                                        <th style={{ padding: '15px' }}>Class Name</th>
                                        <th style={{ padding: '15px' }}>Count</th>
                                        <th style={{ padding: '15px' }}>Percentage</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {datasetStats.class_stats.map((stat, i) => (
                                        <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                            <td style={{ padding: '15px', fontWeight: 'bold' }}>{stat.name}</td>
                                            <td style={{ padding: '15px' }}>{stat.count}</td>
                                            <td style={{ padding: '15px' }}>{stat.percentage}%</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {datasetStats.empty_count > 0 && (
                            <div className="glass section-card" style={{ marginTop: '30px' }}>
                                <div className="section-title" style={{ fontSize: '1.2rem', color: '#ef4444', marginBottom: '15px' }}>Images with No Detections</div>
                                <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                                    {datasetStats.empty_images.map((name, i) => (
                                        <span key={i} className="badge" style={{ background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                                            {name}
                                        </span>
                                    ))}
                                </div>
                                <div style={{ marginTop: '25px', display: 'flex', flexDirection: 'column', gap: '15px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                                        <button
                                            className="btn btn-secondary"
                                            style={{ borderColor: '#ef4444', color: '#ef4444' }}
                                            onClick={handleExtractEmpty}
                                            disabled={isTaskRunning}
                                        >
                                            📦 Extract Empty Images to a Folder
                                        </button>
                                    </div>
                                    <ProgressBar progress={taskProgress} type="extracting_empty" />
                                </div>
                            </div>
                        )}
                    </>
                )}
            </div>
        );
    };

    return (
        <div className="app-wrapper">
            {(!projectConfig) && (
                <div style={{ position: 'absolute', inset: 0, zIndex: 100, background: '#111827' }}>
                    <ProjectLanding
                        onCreateProject={handleCreateProject}
                        onLoadProject={handleLoadProject}
                        onBrowse={handleLandingBrowse}
                    />
                    {showFileBrowser && <FileBrowserModal />}
                </div>
            )}
            <aside className="sidebar">
                <div className="sidebar-logo">
                    <div className="logo-icon">Y</div>
                    YOLO Master
                </div>

                <nav className="nav-links">
                    {navItems.map(item => (
                        <div
                            key={item.id}
                            className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
                            onClick={() => {
                                if (item.id === 'stats') {
                                    if (labelResult?.labeled_dir) {
                                        fetchStats();
                                    }
                                    setActiveTab('stats');
                                } else {
                                    setActiveTab(item.id);
                                }

                                if (item.id === 'dataset_gen' || item.id === 'processing') {
                                    fetchDatasets();
                                }
                            }}
                        >
                            <span className="nav-icon">{item.icon}</span>
                            <span>{item.label}</span>
                        </div>
                    ))}
                </nav>

                <div style={{ marginTop: 'auto', padding: '20px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', fontSize: '0.8rem' }}>
                    <div style={{ color: 'var(--text-muted)', marginBottom: '10px' }}>Task Status</div>
                    {isTaskRunning ? (
                        <div>
                            <div style={{ color: 'var(--primary)', fontWeight: 'bold' }}>{taskProgress?.status.toUpperCase()}</div>
                            <div style={{ opacity: 0.7, marginTop: '4px' }}>{taskProgress?.message}</div>
                        </div>
                    ) : (
                        <div style={{ opacity: 0.5 }}>System Idle</div>
                    )}
                </div>
            </aside>

            <main className="main-content">
                {notification && (
                    <div className="glass" style={{
                        position: 'fixed', top: '20px', right: '40px', padding: '15px 25px',
                        backgroundColor: '#10b981', color: 'white', zIndex: 1000,
                        boxShadow: '0 4px 12px rgba(0,0,0,0.2)', border: 'none'
                    }}>
                        {notification}
                    </div>
                )}

                <FileBrowserModal />
                <LightboxModal />

                <div className="container">
                    {activeTab === 'project_home' && (
                        <section className="glass section-card" style={{ maxWidth: '800px' }}>
                            <div className="section-title">Project Overview</div>

                            {projectConfig ? (
                                <div className="stats-card glass" style={{ marginTop: '20px', textAlign: 'left', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                                        <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{projectConfig.name || 'Unnamed Project'}</div>
                                        <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>{datasetPath}</div>
                                    </div>

                                    <div className="config-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                                        <div className="p-card" style={{ background: 'rgba(0,0,0,0.2)', padding: '15px', borderRadius: '8px' }}>
                                            <div style={{ fontSize: '0.8rem', color: 'var(--accent)', marginBottom: '5px' }}>Raw Images</div>
                                            <code style={{ fontSize: '0.9rem' }}>{projectPaths?.raw || 'N/A'}</code>
                                        </div>
                                        <div className="p-card" style={{ background: 'rgba(0,0,0,0.2)', padding: '15px', borderRadius: '8px' }}>
                                            <div style={{ fontSize: '0.8rem', color: 'var(--accent)', marginBottom: '5px' }}>Processed Images</div>
                                            <code style={{ fontSize: '0.9rem' }}>{projectPaths?.processed || projectConfig.processed_dir || 'N/A'}</code>
                                        </div>
                                        <div className="p-card" style={{ background: 'rgba(0,0,0,0.2)', padding: '15px', borderRadius: '8px' }}>
                                            <div style={{ fontSize: '0.8rem', color: 'var(--accent)', marginBottom: '5px' }}>Annotations</div>
                                            <code style={{ fontSize: '0.9rem' }}>{projectPaths?.annotations || 'N/A'}</code>
                                        </div>
                                        <div className="p-card" style={{ background: 'rgba(0,0,0,0.2)', padding: '15px', borderRadius: '8px' }}>
                                            <div style={{ fontSize: '0.8rem', color: 'var(--accent)', marginBottom: '5px' }}>Labeled Datasets Root</div>
                                            <code style={{ fontSize: '0.9rem' }}>{projectPaths?.labeled || projectConfig.labeled_dir || 'N/A'}</code>
                                        </div>
                                    </div>

                                    <h4 style={{ marginTop: '20px', marginBottom: '10px' }}>Classes</h4>
                                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                                        {projectConfig.classes?.map((c, i) => (
                                            <span key={i} style={{ padding: '5px 10px', background: 'rgba(99, 102, 241, 0.2)', borderRadius: '20px', fontSize: '0.85rem', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                                                {c}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            ) : (
                                <div>No Project Loaded</div>
                            )}
                        </section>
                    )}

                    {activeTab === 'dataset_gen' && (
                        <section className="glass section-card" style={{ maxWidth: '600px' }}>
                            <div className="section-title">Generate Dataset</div>
                            <p style={{ opacity: 0.7, marginBottom: '20px' }}>Create a YOLO dataset by merging processed images with their current annotations.</p>

                            <div className="input-group">
                                <label>Dataset Name</label>
                                <input
                                    type="text"
                                    value={newDatasetName}
                                    onChange={(e) => setNewDatasetName(e.target.value)}
                                    placeholder="e.g. dataset_v1"
                                />
                            </div>

                            <div className="input-group">
                                <label>Strategy</label>
                                <select disabled className="input">
                                    <option>Use All Available Pairs</option>
                                </select>
                            </div>

                            <button
                                className="btn btn-primary"
                                style={{ marginTop: '10px' }}
                                onClick={async () => {
                                    try {
                                        setIsTaskRunning(true);
                                        setTaskProgress({ status: 'creating_dataset', message: 'Creating dataset...', current: 0, total: 100 });
                                        const res = await axios.post(`${API_BASE}/dataset/create`, {
                                            project_path: datasetPath,
                                            name: newDatasetName,
                                            strategy: 'all'
                                        });
                                        showNotification(`Dataset created with ${res.data.count} images`);
                                        setIsTaskRunning(false);
                                        setTaskProgress(null);
                                        fetchDatasets(); // Refresh list
                                    } catch (err) {
                                        showNotification('Error creating dataset: ' + (err.response?.data?.detail || err.message));
                                        setIsTaskRunning(false);
                                    }
                                }}
                                disabled={isTaskRunning}
                            >
                                Generate Dataset
                            </button>
                        </section>
                    )}

                    {activeTab === 'dataset_gen' && (
                        <div style={{ marginTop: '30px' }}>
                            <h3 className="section-title">Existing Datasets</h3>
                            <div className="dataset-list" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '20px' }}>
                                {existingDatasets.map((ds, idx) => (
                                    <div key={idx} className="glass card" style={{ padding: '15px' }}>
                                        <div style={{ fontWeight: 'bold', fontSize: '1.1rem' }}>{ds.name}</div>
                                        <div style={{ fontSize: '0.9rem', opacity: 0.7, marginTop: '5px' }}>
                                            Images: {ds.image_count}
                                        </div>
                                        <div style={{ fontSize: '0.8rem', opacity: 0.5, marginTop: '5px' }}>
                                            {new Date(ds.created_at * 1000).toLocaleString()}
                                        </div>
                                        <div style={{ marginTop: '10px', fontSize: '0.8rem', wordBreak: 'break-all', opacity: 0.6 }}>
                                            {ds.path}
                                        </div>
                                    </div>
                                ))}
                                {existingDatasets.length === 0 && (
                                    <div style={{ opacity: 0.5, gridColumn: '1 / -1', textAlign: 'center', padding: '20px' }}>
                                        No datasets found in this project.
                                    </div>
                                )}
                            </div>
                        </div>
                    )}


                    {activeTab === 'annotation' && (
                        <section className="glass section-card" style={{ height: 'calc(100vh - 150px)', overflow: 'hidden', padding: '10px' }}>
                            <AnnotationTool
                                datasetPath={annotationPath}
                                onPathChange={setAnnotationPath}
                                samModelPath={samModelPath}
                                setSamModelPath={setSamModelPath}
                                onBrowse={openFileBrowser}
                                jumpToImageName={jumpToImageName}
                                onJumpComplete={() => setJumpToImageName(null)}
                                onSave={() => setCacheBuster(Date.now())}
                            />
                        </section>
                    )}

                    {activeTab === 'processing' && (
                        <section className="glass section-card">
                            <div className="section-title">Data Processing: Merge Datasets</div>
                            <p style={{ opacity: 0.7, marginBottom: '25px', fontSize: '0.9rem' }}>
                                Consolidate multiple labeled datasets into one. Class labels will be merged and remapped automatically.
                            </p>

                            <div className="input-group">
                                <label>Source Datasets (folders containing data.yaml, images, and labels)</label>
                                {mergeSources.map((path, idx) => (
                                    <div key={idx} style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                                        <input
                                            type="text"
                                            placeholder="/path/to/labeled/dataset"
                                            value={path}
                                            onChange={(e) => {
                                                const next = [...mergeSources];
                                                next[idx] = e.target.value;
                                                setMergeSources(next);
                                            }}
                                            style={{ flex: 1 }}
                                        />
                                        <button className="browse-btn" onClick={() => openFileBrowser('merge_source', 'dir', idx)}>
                                            Browse
                                        </button>
                                        <button
                                            className="browse-btn"
                                            style={{ color: '#ef4444' }}
                                            onClick={() => removeMergeSource(idx)}
                                            disabled={mergeSources.length === 1}
                                        >
                                            ✕
                                        </button>
                                    </div>
                                ))}
                                <button
                                    className="btn"
                                    style={{ background: 'rgba(240, 239, 239, 0.81)', padding: '10px', fontSize: '0.85rem' }}
                                    onClick={addMergeSource}
                                >
                                    + Add Another Dataset
                                </button>
                            </div>

                            <div className="input-group" style={{ marginTop: '30px' }}>
                                <label>Output Consolidated Dataset Path</label>
                                <div style={{ display: 'flex', gap: '10px' }}>
                                    <input
                                        type="text"
                                        placeholder="/path/to/merged/output"
                                        value={mergeOutput}
                                        onChange={(e) => setMergeOutput(e.target.value)}
                                        style={{ flex: 1 }}
                                    />
                                    <button className="browse-btn" onClick={() => openFileBrowser('merge_output', 'dir')}>
                                        Browse
                                    </button>
                                </div>
                            </div>

                            <div style={{ marginTop: '40px' }}>
                                <button
                                    className="btn btn-primary"
                                    onClick={handleMerge}
                                    disabled={isTaskRunning}
                                    style={{ width: '100%', padding: '15px' }}
                                >
                                    🚀 Start Merging Datasets
                                </button>
                                <ProgressBar progress={taskProgress} type="merging" />
                            </div>

                            {/* Extraction Section */}
                            <div style={{ marginTop: '50px', paddingTop: '40px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                                <div className="section-title">Extraction: Extract Images by Class</div>
                                <p style={{ opacity: 0.7, marginBottom: '25px', fontSize: '0.9rem' }}>
                                    Search a labeled dataset and extract all images containing specific classes into a new folder.
                                </p>

                                <div className="input-group">
                                    <label>Source Labeled Dataset Path</label>
                                    <div style={{ display: 'flex', gap: '10px' }}>
                                        <input
                                            type="text"
                                            placeholder="/path/to/labeled/dataset"
                                            value={extractSource}
                                            onChange={(e) => setExtractSource(e.target.value)}
                                            style={{ flex: 1 }}
                                        />
                                        <button className="browse-btn" onClick={() => openFileBrowser('extract_source', 'dir')}>
                                            Browse
                                        </button>
                                        <button className="btn btn-primary" onClick={handleFetchClasses} style={{ padding: '0 15px' }}>
                                            Fetch Classes
                                        </button>
                                    </div>
                                </div>

                                {availableClasses.length > 0 && (
                                    <div className="input-group" style={{ marginTop: '20px' }}>
                                        <label>Select Classes to Extract</label>
                                        <div style={{
                                            display: 'grid',
                                            gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
                                            gap: '10px',
                                            maxHeight: '200px',
                                            overflowY: 'auto',
                                            padding: '15px',
                                            background: 'rgba(255,255,255,0.03)',
                                            borderRadius: '8px'
                                        }}>
                                            {availableClasses.map((cls, i) => (
                                                <label key={i} style={{
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: '10px',
                                                    cursor: 'pointer',
                                                    fontSize: '0.9rem',
                                                    padding: '5px',
                                                    background: selectedClasses.includes(cls) ? 'rgba(79, 70, 229, 0.1)' : 'transparent',
                                                    borderRadius: '4px'
                                                }}>
                                                    <input
                                                        type="checkbox"
                                                        checked={selectedClasses.includes(cls)}
                                                        onChange={() => toggleClass(cls)}
                                                    />
                                                    {cls}
                                                </label>
                                            ))}
                                        </div>
                                        <div style={{ marginTop: '10px', fontSize: '0.8rem', opacity: 0.6 }}>
                                            {selectedClasses.length} classes selected
                                        </div>
                                    </div>
                                )}

                                <div className="input-group" style={{ marginTop: '30px' }}>
                                    <label>Output Extraction Path</label>
                                    <div style={{ display: 'flex', gap: '10px' }}>
                                        <input
                                            type="text"
                                            placeholder="/path/to/extract/to"
                                            value={extractOutput}
                                            onChange={(e) => setExtractOutput(e.target.value)}
                                            style={{ flex: 1 }}
                                        />
                                        <button className="browse-btn" onClick={() => openFileBrowser('extract_output', 'dir')}>
                                            Browse
                                        </button>
                                    </div>
                                </div>

                                <div style={{ marginTop: '40px' }}>
                                    <button
                                        className="btn btn-primary"
                                        onClick={handleExtract}
                                        disabled={isTaskRunning || selectedClasses.length === 0}
                                        style={{ width: '100%', padding: '15px' }}
                                    >
                                        📦 Extract Matching Images
                                    </button>
                                    <ProgressBar progress={taskProgress} type="extracting" />

                                    {taskProgress?.status === 'idle' && taskProgress?.result?.extracted_count !== undefined && (
                                        <div className="glass" style={{
                                            marginTop: '20px',
                                            padding: '15px',
                                            borderLeft: '4px solid #10b981',
                                            background: 'rgba(16, 185, 129, 0.05)'
                                        }}>
                                            <div style={{ fontWeight: 'bold', color: '#10b981', marginBottom: '5px' }}>Extraction Successful!</div>
                                            <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>
                                                Extracted <strong>{taskProgress.result.extracted_count}</strong> images out of {taskProgress.result.total_scanned} scanned.
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Split Dataset Section */}
                            <div style={{ marginTop: '50px', paddingTop: '40px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                                <div className="section-title">Split: Split Dataset into Parcels</div>
                                <p style={{ opacity: 0.7, marginBottom: '25px', fontSize: '0.9rem' }}>
                                    Split a dataset into multiple equal parts.
                                </p>

                                <div className="input-group">
                                    <label>Dataset to Split</label>
                                    <div style={{ display: 'flex', gap: '10px' }}>
                                        <input
                                            type="text"
                                            placeholder="/path/to/dataset"
                                            value={splitInput}
                                            onChange={(e) => setSplitInput(e.target.value)}
                                            style={{ flex: 1 }}
                                        />
                                        <button className="browse-btn" onClick={() => openFileBrowser('split_input', 'dir')}>
                                            Browse
                                        </button>
                                    </div>
                                </div>

                                <div className="input-group">
                                    <label>Number of Splits</label>
                                    <input
                                        type="number"
                                        min="2"
                                        value={splitCount}
                                        onChange={(e) => setSplitCount(e.target.value)}
                                        style={{ width: '100px' }}
                                    />
                                </div>

                                <div style={{ marginTop: '30px' }}>
                                    <button
                                        className="btn btn-primary"
                                        onClick={handleSplit}
                                        disabled={isTaskRunning}
                                        style={{ width: '100%', padding: '15px' }}
                                    >
                                        ✂️ Split Dataset
                                    </button>
                                    <ProgressBar progress={taskProgress} type="splitting" />

                                    {taskProgress?.status === 'idle' && taskProgress?.result?.processed_count !== undefined && (
                                        <div className="glass" style={{
                                            marginTop: '20px',
                                            padding: '15px',
                                            borderLeft: '4px solid #10b981',
                                            background: 'rgba(16, 185, 129, 0.05)'
                                        }}>
                                            <div style={{ fontWeight: 'bold', color: '#10b981', marginBottom: '5px' }}>Split Successful!</div>
                                            <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>
                                                Processed <strong>{taskProgress.result.processed_count}</strong> images.
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </section>
                    )}

                    {activeTab === 'preprocess' && (
                        <section className="glass section-card">
                            <div className="section-title">Step 2: Pre-processing Configuration</div>
                            <div style={{ marginBottom: '15px', color: '#a1a1aa' }}>
                                Source: <span style={{ fontFamily: 'monospace', color: '#e4e4e7' }}>{projectPaths?.raw || datasetPath}</span>
                            </div>
                            <div className="grid" style={{ gridTemplateColumns: '1fr 350px', gap: '40px' }}>
                                <div>
                                    {sampleImage ? (
                                        <div className="crop-container" style={{ border: '1px solid var(--border-color)', borderRadius: '12px', background: '#000' }}>
                                            <ReactCrop
                                                crop={crop}
                                                onChange={c => setCrop(c)}
                                                onComplete={c => setCompletedCrop(c)}
                                            >
                                                <img
                                                    src={sampleImage.sample_url}
                                                    style={{ maxWidth: '100%', display: 'block' }}
                                                    onLoad={(e) => setImgRef(e.currentTarget)}
                                                />
                                            </ReactCrop>
                                        </div>
                                    ) : (
                                        <div style={{ padding: '80px 40px', textAlign: 'center', opacity: 0.5, border: '2px dashed var(--border-color)', borderRadius: '16px' }}>
                                            Select a dataset first to configure cropping
                                        </div>
                                    )}
                                </div>

                                <div>
                                    <div className="input-group">
                                        <label>Resize dimensions</label>
                                        <div style={{ display: 'flex', gap: '10px' }}>
                                            <input type="number" value={resizeWidth} onChange={e => setResizeWidth(e.target.value)} placeholder="W" />
                                            <input type="number" value={resizeHeight} onChange={e => setResizeHeight(e.target.value)} placeholder="H" />
                                        </div>
                                    </div>

                                    <div className="stats-card" style={{ marginBottom: '20px' }}>
                                        <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Current Crop (Original px)</label>
                                        <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', gap: '15px', marginTop: '15px' }}>
                                            <div className="input-group" style={{ margin: 0 }}>
                                                <label style={{ fontSize: '0.7rem' }}>X</label>
                                                <input
                                                    type="number"
                                                    value={scaledDisplay?.x || 0}
                                                    onChange={e => updateManualCrop('x', e.target.value)}
                                                    style={{ padding: '8px' }}
                                                />
                                            </div>
                                            <div className="input-group" style={{ margin: 0 }}>
                                                <label style={{ fontSize: '0.7rem' }}>Y</label>
                                                <input
                                                    type="number"
                                                    value={scaledDisplay?.y || 0}
                                                    onChange={e => updateManualCrop('y', e.target.value)}
                                                    style={{ padding: '8px' }}
                                                />
                                            </div>
                                            <div className="input-group" style={{ margin: 0 }}>
                                                <label style={{ fontSize: '0.7rem' }}>Width</label>
                                                <input
                                                    type="number"
                                                    value={scaledDisplay?.width || 0}
                                                    onChange={e => updateManualCrop('width', e.target.value)}
                                                    style={{ padding: '8px' }}
                                                />
                                            </div>
                                            <div className="input-group" style={{ margin: 0 }}>
                                                <label style={{ fontSize: '0.7rem' }}>Height</label>
                                                <input
                                                    type="number"
                                                    value={scaledDisplay?.height || 0}
                                                    onChange={e => updateManualCrop('height', e.target.value)}
                                                    style={{ padding: '8px' }}
                                                />
                                            </div>
                                        </div>
                                    </div>

                                    <button className="btn btn-primary" style={{ width: '100%' }} onClick={handlePreprocess} disabled={(!datasetInfo && !projectPaths) || isTaskRunning}>
                                        {isTaskRunning && taskProgress?.status === 'processing' ? 'Processing...' : 'Run Pre-processing'}
                                    </button>
                                    <ProgressBar progress={taskProgress} type="processing" />


                                </div>
                            </div>
                        </section>
                    )}

                    {activeTab === 'labeling' && (
                        <section className="glass section-card" style={labelingMode === 'single' ? { maxWidth: '100%', height: 'calc(100vh - 150px)', overflowY: 'auto' } : { maxWidth: '700px' }}>
                            <div className="section-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                                <span>Step 3: AI Auto-Labeling</span>
                                <div style={{ display: 'flex', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', padding: '4px', border: '1px solid var(--border-color)' }}>
                                    <button
                                        className="btn"
                                        style={{
                                            padding: '6px 15px',
                                            fontSize: '0.8rem',
                                            background: labelingMode === 'batch' ? 'var(--primary)' : 'transparent',
                                            border: 'none',
                                            color: 'white',
                                            borderRadius: '5px'
                                        }}
                                        onClick={() => setLabelingMode('batch')}
                                    >
                                        Batch Process
                                    </button>
                                    <button
                                        className="btn"
                                        style={{
                                            padding: '6px 15px',
                                            fontSize: '0.8rem',
                                            background: labelingMode === 'single' ? 'var(--primary)' : 'transparent',
                                            border: 'none',
                                            color: 'white',
                                            borderRadius: '5px'
                                        }}
                                        onClick={() => setLabelingMode('single')}
                                    >
                                        Single Image
                                    </button>
                                </div>
                            </div>

                            <div className="input-group">
                                <label>YOLO Model Path (.pt)</label>
                                <div style={{ display: 'flex', gap: '10px' }}>
                                    <input
                                        type="text"
                                        placeholder="/path/to/model.pt"
                                        value={modelPath}
                                        onChange={(e) => setModelPath(e.target.value)}
                                        style={{ flex: 1 }}
                                    />
                                    <button className="browse-btn" onClick={() => openFileBrowser('model', 'file')}>
                                        Browse
                                    </button>
                                </div>
                            </div>

                            <div className="input-group">
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <label>Confidence Threshold</label>
                                    <span style={{ color: 'var(--primary)', fontWeight: 'bold' }}>{confidence}</span>
                                </div>
                                <input
                                    type="range" min="0.1" max="1.0" step="0.05"
                                    value={confidence}
                                    onChange={e => setConfidence(e.target.value)}
                                    style={{ width: '100%', marginTop: '10px' }}
                                />
                            </div>

                            <div className="stats-card" style={{ marginBottom: '25px', background: 'rgba(255,255,255,0.03)' }}>
                                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Source Directory</label>
                                <div style={{ fontSize: '0.9rem', marginTop: '5px', wordBreak: 'break-all', fontFamily: 'monospace' }}>
                                    {lastProcessedDir ? lastProcessedDir : datasetPath || 'Not selected'}
                                </div>
                            </div>

                            {labelingMode === 'batch' ? (
                                <>
                                    <button className="btn btn-primary" style={{ height: '50px', fontSize: '1rem' }} onClick={handleAutoLabel} disabled={(!datasetInfo && !projectPaths) || isTaskRunning}>
                                        {isTaskRunning && taskProgress?.status === 'labeling' ? 'Labeling in Progress...' : '⚡ Start Auto-Labeling'}
                                    </button>

                                    <ProgressBar progress={taskProgress} type="labeling" />

                                    {labelResult && (
                                        <div className="stats-card" style={{ marginTop: '30px', borderLeft: '4px solid var(--accent)' }}>
                                            <div style={{ color: 'var(--accent)', fontWeight: 'bold', marginBottom: '15px' }}>✓ Process Complete</div>
                                            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.9rem' }}>
                                                <div><strong>Output:</strong> {labelResult.labeled_dir}</div>
                                                <div><strong>Config:</strong> {labelResult.yaml_path}</div>
                                                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '5px' }}>
                                                    {labelResult.classes && labelResult.classes.map((c, i) => (
                                                        <span key={i} className="badge">{c}</span>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </>
                            ) : (
                                <div className="single-label-container" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', minHeight: '400px' }}>
                                    {/* Left: Raw Image */}
                                    <div className="glass" style={{ padding: '15px', display: 'flex', flexDirection: 'column' }}>
                                        <div style={{ marginBottom: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                <span style={{ fontWeight: 'bold' }}>Raw Image</span>
                                                {labelingImages.length > 0 && (
                                                    <span style={{ fontSize: '0.8rem', opacity: 0.6 }}>
                                                        {currentSingleImageIndex + 1} / {labelingImages.length}
                                                    </span>
                                                )}
                                            </div>
                                            <div className="input-group" style={{ margin: 0 }}>
                                                <input
                                                    type="text"
                                                    placeholder="Search image name..."
                                                    value={imageSearchQuery}
                                                    onChange={handleImageSearch}
                                                    style={{ padding: '6px', fontSize: '0.9rem', width: '100%' }}
                                                />
                                            </div>
                                        </div>

                                        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', overflow: 'hidden', minHeight: '300px', position: 'relative' }}>
                                            {labelingImages.length > 0 ? (
                                                <img
                                                    src={`http://localhost:8000/static/labeling_source/${labelingImages[currentSingleImageIndex]?.name}?t=${cacheBuster}`}
                                                    style={{ maxWidth: '100%', maxHeight: '400px', objectFit: 'contain' }}
                                                />
                                            ) : (
                                                <div style={{ opacity: 0.5 }}>No images found</div>
                                            )}
                                        </div>

                                        <div style={{ display: 'flex', gap: '10px', marginTop: '15px', justifyContent: 'center' }}>
                                            <button
                                                className="btn"
                                                onClick={() => navigateSingleImage(-1)}
                                                disabled={currentSingleImageIndex === 0}
                                            >
                                                Previous (←)
                                            </button>
                                            <div style={{ flex: 1, textAlign: 'center', fontSize: '0.8rem', fontFamily: 'monospace', alignSelf: 'center' }}>
                                                {labelingImages[currentSingleImageIndex]?.name || '-'}
                                            </div>
                                            <button
                                                className="btn"
                                                onClick={() => navigateSingleImage(1)}
                                                disabled={currentSingleImageIndex === labelingImages.length - 1}
                                            >
                                                Next (→)
                                            </button>
                                        </div>
                                    </div>

                                    {/* Right: Result */}
                                    <div className="glass" style={{ padding: '15px', display: 'flex', flexDirection: 'column' }}>
                                        <div style={{ marginBottom: '10px', fontWeight: 'bold' }}>Labeling Result</div>

                                        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', overflow: 'hidden', minHeight: '300px' }}>
                                            {singleLabelLoading ? (
                                                <div className="spinner"></div>
                                            ) : singleLabelResult && singleLabelResult.masked_url ? (
                                                <img
                                                    src={`http://localhost:8000${singleLabelResult.masked_url}?t=${cacheBuster}`}
                                                    style={{ maxWidth: '100%', maxHeight: '400px', objectFit: 'contain' }}
                                                />
                                            ) : (
                                                <div style={{ opacity: 0.3, textAlign: 'center' }}>
                                                    <div>Result will appear here</div>
                                                </div>
                                            )}
                                        </div>

                                        <div style={{ marginTop: '15px' }}>
                                            <button
                                                className="btn btn-primary"
                                                style={{ width: '100%' }}
                                                onClick={handleAutoLabelSingle}
                                                disabled={singleLabelLoading || !labelingImages[currentSingleImageIndex]}
                                            >
                                                {singleLabelLoading ? 'Processing...' : '⚡ Auto-Label This Image'}
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </section>
                    )}

                    {activeTab === 'verification' && (
                        <section className="glass section-card">
                            <div className="section-title">Step 4: Quality Verification</div>
                            <p style={{ color: 'var(--text-muted)', marginBottom: '30px' }}>
                                Preview labeled results with confidence scores and bounding boxes.
                            </p>
                            <VerificationGallery onJumpToAnnotation={jumpToAnnotation} />
                        </section>
                    )}

                    {activeTab === 'stats' && (
                        <section className="glass section-card">
                            <div className="section-title">Step 5: Dataset Insights</div>
                            {statsLoading ? (
                                <div style={{ padding: '100px', textAlign: 'center' }}>
                                    <div className="spinner" style={{ marginBottom: '20px' }}></div>
                                    <p>Generating comprehensive analytics...</p>
                                </div>
                            ) : (
                                <StatsView />
                            )}
                        </section>
                    )}
                </div>
            </main >
        </div >
    );
}

export default App;
