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
import ProgressBar from './components/ProgressBar';
import LightboxModal from './components/LightboxModal';
import VerificationGallery from './components/VerificationGallery';
import StatsView from './components/StatsView';
import DatasetSampling from './components/DatasetSampling';
import TrainingView from './components/TrainingView';
import DatasetToolsPage from './pages/DatasetToolsPage';

const API_BASE = '/api';
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
    const [dynamicRenderPath, setDynamicRenderPath] = useState(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [projectConfig, setProjectConfig] = useState(null);
    const [projectPaths, setProjectPaths] = useState(null);
    const [isStandalone, setIsStandalone] = useState(false);
    const [landingCallback, setLandingCallback] = useState(null);
    const [trainingCallback, setTrainingCallback] = useState(null);
    const MASKED_LIMIT = 20;

    // Lightbox & Stats State
    const [selectedLightboxImage, setSelectedLightboxImage] = useState(null);
    const [datasetStats, setDatasetStats] = useState(null);
    const [statsLoading, setStatsLoading] = useState(false);

    // Cache buster for images
    const [cacheBuster, setCacheBuster] = useState(Date.now());
    const [verificationScroll, setVerificationScroll] = useState(0);

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
        { type: 'header', label: 'TRAINING' },
        { id: 'training', label: 'Train Model', icon: '🏋️' },
    ]);
    const [newDatasetName, setNewDatasetName] = useState('dataset_v1');
    const [datasetStrategy, setDatasetStrategy] = useState('all');
    const [datasetSplitRatios, setDatasetSplitRatios] = useState({ train: 70, val: 20, test: 10 });


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

    // Rebalance State
    const [rebalanceInput, setRebalanceInput] = useState('');
    const [rebalanceStats, setRebalanceStats] = useState(null);
    const [newRebalancePcts, setNewRebalancePcts] = useState({ train: 70, val: 20, test: 10 });
    const [deleteOriginalRebalance, setDeleteOriginalRebalance] = useState(false);

    // Flatten State
    const [flattenInput, setFlattenInput] = useState('');

    // Verification Filtering
    const [filterClasses, setFilterClasses] = useState([]);

    // Stats State
    const [statsPath, setStatsPath] = useState('');
    const [selectedSplit, setSelectedSplit] = useState('all');
    const [datasetSplits, setDatasetSplits] = useState([]);

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
            let res;
            if (data.is_split_mode) {
                res = await axios.post(`${API_BASE}/project/create_from_split`, data);
                showNotification('Project created from split dataset successfully!');
            } else {
                res = await axios.post(`${API_BASE}/project/create`, data);
                showNotification('Project created successfully!');
            }
            await handleLoadProject(res.data.path);
        } catch (err) {
            console.error(err);
            alert("Error creating project: " + (err.response?.data?.detail || err.message));
            throw err;
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

    const saveProjectConfig = async (updates) => {
        if (!datasetPath) return;
        try {
            await axios.post(`${API_BASE}/project/config`, {
                path: datasetPath,
                updates: updates
            });
            // Update local config state partially to avoid reload
            setProjectConfig(prev => ({ ...prev, ...updates }));
        } catch (err) {
            console.error("Failed to save config:", err);
        }
    };

    // Auto-save SAM model path when it changes (debounced)
    useEffect(() => {
        if (!projectConfig || !samModelPath) return;
        if (samModelPath === projectConfig.sam_model_path) return;

        const timer = setTimeout(() => {
            console.log("Auto-saving SAM model path...");
            saveProjectConfig({ sam_model_path: samModelPath });
        }, 1000);

        return () => clearTimeout(timer);
    }, [samModelPath, projectConfig]);

    // Consolidated Path Selector for Annotation
    useEffect(() => {
        if (!datasetPath) {
            setAnnotationPath('');
            return;
        }

        if (projectPaths?.processed) {
            setAnnotationPath(projectPaths.processed);
        } else {
            setAnnotationPath(datasetPath);
        }
    }, [datasetPath, projectPaths]);

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
            // Data Inspection logic
            console.log("Trace Verification: datasetPath=", datasetPath, "masked=", !!projectPaths?.masked, "hasMasks=", hasMasks, "hasLabels=", hasLabels, "dynamic=", !!dynamicRenderPath);

            // 1. Try Loading Static Masked Images
            // Fix: Only auto-load static masks if they actually exist (hasMasks)
            if (projectPaths.masked && hasMasks) {
                console.log("Trace: Attempting Static Load");
                // Fix: Only load if path changed or empty (Prevent Reset)
                // AND ensure we aren't currently using dynamic rendering (fallback)
                if (!dynamicRenderPath && (projectPaths.masked !== maskedPath || maskedImages.length === 0)) {
                    console.log("Trace: Triggering handleLoadMaskedImages (Static)");
                    handleLoadMaskedImages(projectPaths.masked);
                }
            }
            // 2. If no static masks, but we have labels, force Dynamic Mode
            else if (hasLabels) {
                const imgSource = projectPaths.processed || datasetPath;
                if (imgSource && (imgSource !== maskedPath || maskedImages.length === 0)) {
                    // Only trigger if we haven't already loaded this path
                    setDynamicRenderPath(imgSource);
                    handleLoadMaskedImages(imgSource, 0, false, true);
                }
            }
        }

        if (activeTab === 'stats') {
            fetchDatasets();
        }
    }, [activeTab, projectConfig, projectPaths, hasLabels, hasMasks, dynamicRenderPath]); // Added dependencies

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

    // Load defaults from Project Config
    useEffect(() => {
        if (projectConfig) {
            if (projectConfig.classes && Array.isArray(projectConfig.classes)) {
                setAvailableClasses(projectConfig.classes);
                // Also update filter classes if empty
                // if (filterClasses.length === 0) setFilterClasses(projectConfig.classes);
            }
            // Load auto-label model
            const mPath = projectConfig.model_path || projectConfig.model;
            if (mPath) {
                setModelPath(mPath);
            }
            // Load SAM model
            if (projectConfig.sam_model_path) {
                setSamModelPath(projectConfig.sam_model_path);
            }
        }
    }, [projectConfig]);

    // (Effect removed, consolidated above)


    const handleLoadMaskedImages = async (path, offset = 0, append = false, isDynamic = false) => {
        try {
            if (offset === 0) {
                if (isDynamic) {
                    setMaskedMountUrl(null);
                } else {
                    const mountRes = await axios.post(`${API_BASE}/mount?name=masked&path=${encodeURIComponent(path)}`);
                    setMaskedMountUrl(mountRes.data.url);
                }
                setMaskedPath(path);
                setCacheBuster(Date.now()); // Update cache buster on new path load
            }

            let url = `${API_BASE}/labeled/images?path=${encodeURIComponent(path)}&limit=${MASKED_LIMIT}&offset=${offset}`;
            if (filterClasses.length > 0) {
                url += `&classes=${encodeURIComponent(filterClasses.join(','))}`;
            }
            if (searchQuery) {
                url += `&search=${encodeURIComponent(searchQuery)}`;
            }
            if (selectedSplit && selectedSplit !== 'all') {
                url += `&split=${encodeURIComponent(selectedSplit)}`;
            }

            const imagesRes = await axios.get(url);
            if (append) {
                setMaskedImages(prev => [...prev, ...imagesRes.data.images]);
            } else {
                setMaskedImages(imagesRes.data.images);
            }
            setTotalMasked(imagesRes.data.total);
            setMaskedOffset(offset);

            // Fallback: If static load returned 0 images, but we have labels, switch to dynamic
            if (!isDynamic && imagesRes.data.total === 0 && hasLabels) {
                const imgSource = projectPaths?.processed || datasetPath;
                if (imgSource) {
                    setDynamicRenderPath(imgSource);
                    setMaskedPath(imgSource); // IMPORTANT: Update maskedPath so filters use this new path
                    // Recursively call with dynamic mode
                    handleLoadMaskedImages(imgSource, 0, false, true);
                    showNotification('No static masks found. Switched to dynamic visualization.');
                }
            }
        } catch (err) {
            console.error('Error loading masked images:', err);
        }
    };

    const handlePageChange = (newOffset) => {
        const path = maskedPath || labelResult?.masked_dir;
        if (path) {
            handleLoadMaskedImages(path, newOffset, false, !!dynamicRenderPath);
            // Reset scroll to top when changing pages
            setVerificationScroll(0);
            const scrollContainer = document.querySelector('.main-content');
            if (scrollContainer) scrollContainer.scrollTop = 0;
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

    // Reload gallery when split or searched classes change
    useEffect(() => {
        if (maskedPath && activeTab === 'verification') {
            handleLoadMaskedImages(maskedPath, 0, false, !!dynamicRenderPath);
        }
    }, [selectedSplit]);

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
        console.log("Trace: loadDatasetInfo START. Path:", datasetPath);
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
        setDynamicRenderPath(null);
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
            console.log("Trace loadDatasetInfo: path=", datasetPath, "status=", statusRes.data);
            const { processed_dir, labeled_dir, labels_root, has_labels, has_masks, config } = statusRes.data;

            setHasMasks(has_masks);
            setHasLabels(has_labels);
            setProjectConfig(config);

            if (statusRes.data.is_split) {
                setDatasetSplits(['all', 'train', 'valid', 'test']);
            } else {
                setDatasetSplits([]);
            }
            setSelectedSplit('all'); // Always reset split filter for new dataset

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
                    const imgSource = processed_dir || datasetPath;
                    if (imgSource) {
                        setDynamicRenderPath(imgSource);
                        handleLoadMaskedImages(imgSource, 0, false, true);
                        showNotification('Found existing labels. Loaded dynamic visualization.');
                    } else {
                        showNotification('Found existing labels. You can generate masked images in the Verification tab.');
                    }
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
        if (!annotationPath || annotationPath !== datasetPath) {
            setAnnotationPath(datasetPath);
        }

        setJumpToImageName(imageName);
        setActiveTab('annotation');
    };

    const handleLandingBrowse = (cb, type) => {
        setLandingCallback(() => cb);
        openFileBrowser('landing_generic', type);
    };

    const handleTrainingBrowse = (target, type, cb) => {
        setTrainingCallback(() => cb);
        openFileBrowser('training_generic', type);
    };

    const openTrainingBrowser = (type, cb) => {
        setTrainingCallback(() => cb);
        openFileBrowser('training_generic', type === 'model' ? 'file' : 'dir');
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
            else if (browserTarget === 'training_generic' && trainingCallback) trainingCallback(item.path);
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
            else if (browserTarget === 'rebalance_input') {
                setRebalanceInput(browserPath);
                fetchRebalanceStats(browserPath);
            }
            else if (browserTarget === 'flatten_input') setFlattenInput(browserPath);
            else if (browserTarget === 'landing_generic') {
                if (landingCallback) landingCallback(browserPath);
            }
            else if (browserTarget === 'training_generic') {
                if (trainingCallback) trainingCallback(browserPath);
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
                            {browserTarget === 'rebalance_input' && 'Select Dataset to Re-balance'}
                            {browserTarget === 'landing_generic' && 'Select Folder'}
                            {browserTarget === 'training_generic' && 'Select'}

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
            // Fix: Use current 'crop' state, as 'completedCrop' might be stale if user manually edited inputs
            // 'crop' is the single source of truth updated by both Drag and Manual Input
            const scaled = calculateScaledCrop(crop);
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

    const handleExtractEmpty = async (imagesList = null) => {
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
                labeled_root: activeTab === 'stats' ? (statsPath || projectPaths?.annotations) : labelResult?.labeled_dir,
                images_list: Array.isArray(imagesList) ? imagesList : null
            });
            showNotification('Extraction task started...');
        } catch (err) {
            showNotification('Failed to start extraction: ' + (err.response?.data?.detail || err.message));
            setIsTaskRunning(false);
            setTaskProgress(null);
        }
    };

    const fetchRebalanceStats = async (path) => {
        try {
            const res = await axios.post(`${API_BASE}/dataset/split-stats`, { dataset_path: path });
            setRebalanceStats(res.data);
        } catch (err) {
            console.error(err);
            showNotification('Failed to fetch dataset stats');
        }
    };

    const handleRebalance = async () => {
        if (!rebalanceInput) {
            showNotification('Please select a dataset');
            return;
        }

        const total = newRebalancePcts.train + newRebalancePcts.val + newRebalancePcts.test;
        if (total !== 100) {
            showNotification(`Total percentage must be 100% (Current: ${total}%)`);
            return;
        }

        setIsTaskRunning(true);
        setTaskProgress({ status: 'rebalancing', message: 'Re-balancing dataset...', current: 0, total: 100 });

        try {
            const res = await axios.post(`${API_BASE}/dataset/rebalance`, {
                dataset_path: rebalanceInput,
                train_pct: newRebalancePcts.train / 100,
                val_pct: newRebalancePcts.val / 100,
                test_pct: newRebalancePcts.test / 100,
                delete_original: deleteOriginalRebalance
            });

            showNotification(res.data.message);
            // Update stats with new location if moved/changed
            setRebalanceInput(res.data.new_path);
            setRebalanceStats(res.data.stats);

            setIsTaskRunning(false);
            setTaskProgress(null);
        } catch (err) {
            showNotification('Re-balance failed: ' + (err.response?.data?.detail || err.message));
            setIsTaskRunning(false);
            setTaskProgress(null);
        }
    };

    const handleFlatten = async () => {
        if (!flattenInput) {
            showNotification('Please select a dataset to flatten');
            return;
        }

        setIsTaskRunning(true);
        setTaskProgress({ status: 'flattening', message: 'Flattening dataset...', current: 0, total: 100 });
        try {
            await axios.post(`${API_BASE}/dataset/flatten`, {
                dataset_path: flattenInput
            });
            showNotification('Dataset flattening started (or completed)');
        } catch (err) {
            showNotification('Flattening failed: ' + (err.response?.data?.detail || err.message));
            setIsTaskRunning(false);
            setTaskProgress(null);
        }
    };

    const handleGenerateMasks = async () => {
        if (!datasetPath) {
            showNotification('Please load a dataset first.');
            return;
        }

        // Determine labels path: Priority to active labelResult, fallback to project config
        let labelsPath = labelResult?.labeled_dir;

        if (!labelsPath && projectPaths?.annotations) {
            labelsPath = projectPaths.annotations;
        }

        if (!labelsPath) {
            showNotification('No labeled directory found. Please run auto-labeling first.');
            return;
        }

        setIsTaskRunning(true);
        setTaskProgress({ status: 'generating_masks', message: 'Starting mask generation...', current: 0, total: 100 });
        try {
            const res = await axios.post(`${API_BASE}/generate_masked`, {
                dataset_path: datasetPath,
                labels_path: labelsPath
            });
            showNotification('Mask generation started...');

            // If simple state update, set it
            if (!labelResult) {
                setLabelResult({
                    labeled_dir: labelsPath,
                    masked_dir: '', // Unknown yet
                    classes: [],
                    yaml_path: 'Project Config'
                });
                // Also ensure buttons show up by setting flags
                setHasLabels(true);
            }

        } catch (err) {
            showNotification('Failed to start mask generation: ' + (err.response?.data?.detail || err.message));
            setIsTaskRunning(false);
            setTaskProgress(null);
        }
    };

    const handleRefreshGallery = () => {
        if (maskedPath) {
            setCacheBuster(Date.now());
            handleLoadMaskedImages(maskedPath, 0, false, !!dynamicRenderPath);
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
            if (prev.includes(cls)) {
                return prev.filter(c => c !== cls);
            } else {
                return [...prev, cls];
            }
        });
    };

    // Load Dataset Info when path changes
    useEffect(() => {
        if (datasetPath) {
            console.log("Trace: useEffect triggered by datasetPath:", datasetPath);
            loadDatasetInfo();
        }
    }, [datasetPath]);

    // Reload gallery when filter or search changes
    useEffect(() => {
        if (maskedPath) {
            handleLoadMaskedImages(maskedPath, 0, false, !!dynamicRenderPath);
        }
    }, [filterClasses, searchQuery]);








    return (
        <div className="app-wrapper">
            {(!projectConfig && !isStandalone) && (
                <div style={{ position: 'absolute', inset: 0, zIndex: 100, background: '#111827' }}>
                    <ProjectLanding
                        onCreateProject={handleCreateProject}
                        onLoadProject={handleLoadProject}
                        onBrowse={handleLandingBrowse}
                        onSkip={(targetTab = 'training') => {
                            setIsStandalone(true);
                            setActiveTab(targetTab);
                        }}
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
                    {navItems.map(item => {
                        if (item.type === 'header') {
                            return <div key={item.label} className="nav-header">{item.label}</div>;
                        }

                        // Allow Data Inspection even without a project
                        const isVisible = projectConfig || isStandalone || item.id === 'verification';
                        if (!isVisible) return null;

                        return (
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
                                        // If clicking Data Inspection or Manual Annotation on a fresh start, enter standalone mode
                                        if ((item.id === 'verification' || item.id === 'annotation') && !projectConfig) {
                                            setIsStandalone(true);
                                        }
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
                        );
                    })}
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
                <FileBrowserModal />
                <LightboxModal
                    selectedLightboxImage={selectedLightboxImage}
                    onClose={() => setSelectedLightboxImage(null)}
                    maskedMountUrl={maskedMountUrl}
                    dynamicRenderPath={dynamicRenderPath}
                    cacheBuster={cacheBuster}
                />

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
                                <select
                                    className="input"
                                    value={datasetStrategy}
                                    onChange={(e) => setDatasetStrategy(e.target.value)}
                                >
                                    <option value="all">Use All Available Pairs</option>
                                    <option value="split">Split Train/Val/Test</option>
                                </select>
                            </div>

                            {datasetStrategy === 'split' && (
                                <div className="input-group">
                                    <label>Split Ratios (%)</label>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
                                        <div>
                                            <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>Train</span>
                                            <input
                                                type="number"
                                                value={datasetSplitRatios.train}
                                                onChange={(e) => setDatasetSplitRatios({ ...datasetSplitRatios, train: parseInt(e.target.value) || 0 })}
                                            />
                                        </div>
                                        <div>
                                            <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>Val</span>
                                            <input
                                                type="number"
                                                value={datasetSplitRatios.val}
                                                onChange={(e) => setDatasetSplitRatios({ ...datasetSplitRatios, val: parseInt(e.target.value) || 0 })}
                                            />
                                        </div>
                                        <div>
                                            <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>Test</span>
                                            <input
                                                type="number"
                                                value={datasetSplitRatios.test}
                                                onChange={(e) => setDatasetSplitRatios({ ...datasetSplitRatios, test: parseInt(e.target.value) || 0 })}
                                            />
                                        </div>
                                    </div>
                                    {(datasetSplitRatios.train + datasetSplitRatios.val + datasetSplitRatios.test) !== 100 && (
                                        <div style={{ color: '#ef4444', fontSize: '0.8rem', marginTop: '5px' }}>
                                            Total: {datasetSplitRatios.train + datasetSplitRatios.val + datasetSplitRatios.test}% (Must be 100%)
                                        </div>
                                    )}
                                </div>
                            )}

                            <button
                                className="btn btn-primary"
                                style={{ marginTop: '10px' }}
                                disabled={isTaskRunning || (datasetStrategy === 'split' && (datasetSplitRatios.train + datasetSplitRatios.val + datasetSplitRatios.test) !== 100)}
                                onClick={async () => {
                                    try {
                                        setIsTaskRunning(true);
                                        setTaskProgress({ status: 'creating_dataset', message: 'Creating dataset...', current: 0, total: 100 });

                                        const payload = {
                                            project_path: datasetPath,
                                            name: newDatasetName,
                                            strategy: datasetStrategy
                                        };

                                        if (datasetStrategy === 'split') {
                                            payload.split_ratios = [
                                                datasetSplitRatios.train / 100,
                                                datasetSplitRatios.val / 100,
                                                datasetSplitRatios.test / 100
                                            ];
                                        }

                                        const res = await axios.post(`${API_BASE}/dataset/create`, payload);
                                        showNotification(`Dataset created with ${res.data.count} images`);
                                        setIsTaskRunning(false);
                                        setTaskProgress(null);
                                        fetchDatasets(); // Refresh list
                                    } catch (err) {
                                        showNotification('Error creating dataset: ' + (err.response?.data?.detail || err.message));
                                        setIsTaskRunning(false);
                                    }
                                }}
                            >
                                {isTaskRunning && taskProgress?.status === 'creating_dataset' ? '⏳ Generating...' : 'Generate Dataset'}
                            </button>
                            <ProgressBar progress={taskProgress} type="creating_dataset" />
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
                                            {ds.splits && Object.keys(ds.splits).length > 0 && (
                                                <div style={{ marginTop: '5px' }}>
                                                    {Object.entries(ds.splits).map(([k, v]) => (
                                                        <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                                                            <span style={{ textTransform: 'capitalize' }}>{k}:</span>
                                                            <span>{v} ({Math.round(v / ds.image_count * 100)}%)</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
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
                                key={`${annotationPath}-${selectedSplit}`}
                                datasetPath={annotationPath}
                                selectedSplit={selectedSplit}
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
                        <DatasetToolsPage
                            mergeSources={mergeSources}
                            setMergeSources={setMergeSources}
                            addMergeSource={addMergeSource}
                            removeMergeSource={removeMergeSource}
                            mergeOutput={mergeOutput}
                            setMergeOutput={setMergeOutput}
                            handleMerge={handleMerge}
                            extractSource={extractSource}
                            setExtractSource={setExtractSource}
                            availableClasses={availableClasses}
                            selectedClasses={selectedClasses}
                            handleFetchClasses={handleFetchClasses}
                            toggleClass={toggleClass}
                            toggleFilterClass={toggleFilterClass}
                            extractOutput={extractOutput}
                            setExtractOutput={setExtractOutput}
                            handleExtract={handleExtract}
                            rebalanceInput={rebalanceInput}
                            setRebalanceInput={setRebalanceInput}
                            rebalanceStats={rebalanceStats}
                            newRebalancePcts={newRebalancePcts}
                            setNewRebalancePcts={setNewRebalancePcts}
                            deleteOriginalRebalance={deleteOriginalRebalance}
                            setDeleteOriginalRebalance={setDeleteOriginalRebalance}
                            handleRebalance={handleRebalance}
                            fetchRebalanceStats={fetchRebalanceStats}
                            flattenInput={flattenInput}
                            setFlattenInput={setFlattenInput}
                            handleFlatten={handleFlatten}
                            splitInput={splitInput}
                            setSplitInput={setSplitInput}
                            splitCount={splitCount}
                            setSplitCount={setSplitCount}
                            handleSplit={handleSplit}
                            datasetPath={datasetPath}
                            projectConfig={projectConfig}
                            handleLandingBrowse={handleLandingBrowse}
                            isTaskRunning={isTaskRunning}
                            taskProgress={taskProgress}
                            openFileBrowser={openFileBrowser}
                            setIsTaskRunning={setIsTaskRunning}
                            showNotification={showNotification}
                        />
                    )}

                    {
                        activeTab === 'preprocess' && (
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
                                                    onChange={(_, percentCrop) => setCrop(percentCrop)}
                                                    onComplete={(_, percentCrop) => setCompletedCrop(percentCrop)}
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
                        )
                    }

                    {
                        activeTab === 'labeling' && (
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
                                                        src={`/static/labeling_source/${labelingImages[currentSingleImageIndex]?.name}?t=${cacheBuster}`}
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
                                                        src={`${singleLabelResult.masked_url}?t=${cacheBuster}`}
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
                        )
                    }

                    {
                        activeTab === 'verification' && (
                            <section className="glass section-card">
                                <div className="section-title">Step 4: Quality Verification</div>
                                <p style={{ color: 'var(--text-muted)', marginBottom: '30px' }}>
                                    Preview labeled results with confidence scores and bounding boxes.
                                </p>
                                {datasetPath ? (
                                    <VerificationGallery
                                        onJumpToAnnotation={jumpToAnnotation}
                                        availableClasses={availableClasses}
                                        datasetStats={datasetStats}
                                        filterClasses={filterClasses}
                                        setFilterClasses={setFilterClasses}
                                        searchQuery={searchQuery}
                                        setSearchQuery={setSearchQuery}
                                        maskedOffset={maskedOffset}
                                        maskedImages={maskedImages}
                                        totalMasked={totalMasked}
                                        isTaskRunning={isTaskRunning}
                                        handleGenerateMasks={handleGenerateMasks}
                                        handleRefreshGallery={handleRefreshGallery}
                                        toggleFilterClass={toggleFilterClass}
                                        hasLabels={hasLabels}
                                        hasMasks={hasMasks}
                                        taskProgress={taskProgress}
                                        setSelectedLightboxImage={setSelectedLightboxImage}
                                        maskedMountUrl={maskedMountUrl}
                                        cacheBuster={cacheBuster}
                                        handleFilterImage={handleFilterImage}
                                        MASKED_LIMIT={MASKED_LIMIT}
                                        handlePageChange={handlePageChange}
                                        verificationScroll={verificationScroll}
                                        setVerificationScroll={setVerificationScroll}
                                        dynamicRenderPath={dynamicRenderPath}
                                        selectedSplit={selectedSplit}
                                        onSplitChange={setSelectedSplit}
                                        datasetSplits={datasetSplits}
                                    />
                                ) : (
                                    <div style={{ padding: '80px 40px', textAlign: 'center', background: 'rgba(255,255,255,0.02)', borderRadius: '16px', border: '2px dashed var(--border-color)' }}>
                                        <div style={{ fontSize: '1.2rem', marginBottom: '20px', opacity: 0.7 }}>No dataset selected for inspection.</div>
                                        <button className="btn btn-primary" onClick={() => openFileBrowser('dataset', 'dir')}>
                                            📁 Select Dataset Directory
                                        </button>
                                        <p style={{ marginTop: '15px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                                            Choose a YOLO dataset folder (contains images/ and labels/ or train/val/test splits)
                                        </p>
                                    </div>
                                )}
                            </section>
                        )
                    }

                    {
                        activeTab === 'stats' && (
                            <section className="glass section-card">
                                <div className="section-title">Step 5: Dataset Insights</div>
                                {statsLoading ? (
                                    <div style={{ padding: '100px', textAlign: 'center' }}>
                                        <div className="spinner" style={{ marginBottom: '20px' }}></div>
                                        <p>Generating comprehensive analytics...</p>
                                    </div>
                                ) : (
                                    <StatsView
                                        projectPaths={projectPaths}
                                        existingDatasets={existingDatasets}
                                        datasetStats={datasetStats}
                                        statsPath={statsPath}
                                        setStatsPath={setStatsPath}
                                        fetchStats={fetchStats}
                                        handleExtractEmpty={handleExtractEmpty}
                                        isTaskRunning={isTaskRunning}
                                        taskProgress={taskProgress}
                                        datasetPath={datasetPath}
                                        projectConfig={projectConfig}
                                        activeTab={activeTab}
                                        labelResult={labelResult}
                                    />
                                )}
                            </section>
                        )
                    }
                    {
                        activeTab === 'training' && (
                            <div style={{ height: 'calc(100vh - 40px)', overflow: 'hidden' }}>
                                <TrainingView
                                    datasetPath={projectPaths?.processed || datasetPath} // Fallback to datasetPath
                                    onBrowse={handleTrainingBrowse}
                                />
                            </div>
                        )
                    }
                </div >
            </main >
        </div >
    );
}

export default App;
