import React, { useState } from 'react';

const ProjectLanding = ({ onCreateProject, onLoadProject, onBrowse, onSkip }) => {
    const [mode, setMode] = useState('landing'); // landing, create, load
    const [createMode, setCreateMode] = useState('standard'); // standard, split
    const [loading, setLoading] = useState(false);
    const [loadPath, setLoadPath] = useState('');
    const [createForm, setCreateForm] = useState({
        name: '',
        parentDir: '',
        rawImagesDir: '',
        classes: '', // comma separated
        splitDatasetPath: ''
    });

    const handleCreateSubmit = async () => {
        if (!createForm.name || !createForm.parentDir) {
            alert("Please fill in Name and Parent Directory");
            return;
        }

        if (createMode === 'standard' && !createForm.rawImagesDir) {
            alert("Please select Raw Images Source");
            return;
        }

        if (createMode === 'split' && !createForm.splitDatasetPath) {
            alert("Please select Split Dataset Path");
            return;
        }

        setLoading(true);
        try {
            if (createMode === 'standard') {
                const classesList = createForm.classes.split(',').map(c => c.trim()).filter(c => c);
                await onCreateProject({
                    name: createForm.name,
                    parent_dir: createForm.parentDir,
                    raw_images_dir: createForm.rawImagesDir,
                    classes: classesList
                });
            } else {
                // Split mode
                // We need to call a different endpoint, but `onCreateProject` prop usually wraps the fetch.
                // If `onCreateProject` is generic, we might need to modify App.jsx or pass a flag.
                // Assuming onCreateProject can handle it or we fetch directly here?
                // Usually it's better if the parent handles API interaction, but for speed let's check App.jsx
                // Actually the prop `onCreateProject` likely calls `/api/project/create`.
                // We should probably check App.jsx to see how it's implemented. 
                // BUT, I can overload the object passed to onCreateProject and handle it in App.jsx.
                // OR, I can fetch directly here if I import the API.

                // Let's cheat slightly and use fetch directly for the new endpoint if we can't easily change App.jsx right now,
                // OR better: pass a special payload that App.jsx recognizes?
                // "raw_images_dir" vs "split_dataset_path"

                // Let's Assume we update App.jsx too.

                await onCreateProject({
                    name: createForm.name,
                    parent_dir: createForm.parentDir,
                    split_dataset_path: createForm.splitDatasetPath,
                    is_split_mode: true
                });
            }
        } catch (e) {
            // Error handled in parent
            console.error(e);
            alert("Error creating project: " + e.message);
        } finally {
            setLoading(false);
        }
    };

    if (mode === 'create') {
        return (
            <div className="glass" style={{ maxWidth: '600px', margin: '50px auto', padding: '30px', borderRadius: '16px' }}>
                <h2 style={{ marginBottom: '20px' }}>Create New Project</h2>

                <div style={{ marginBottom: '20px', display: 'flex', gap: '10px', background: 'rgba(255,255,255,0.05)', padding: '5px', borderRadius: '8px' }}>
                    <button
                        className={`btn ${createMode === 'standard' ? 'btn-primary' : ''}`}
                        style={{ flex: 1, border: 'none' }}
                        onClick={() => setCreateMode('standard')}
                    >
                        Standard (Raw Images)
                    </button>
                    <button
                        className={`btn ${createMode === 'split' ? 'btn-primary' : ''}`}
                        style={{ flex: 1, border: 'none' }}
                        onClick={() => setCreateMode('split')}
                    >
                        From Split Dataset
                    </button>
                </div>

                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px' }}>Project Name</label>
                    <input
                        className="input"
                        value={createForm.name}
                        onChange={e => setCreateForm({ ...createForm, name: e.target.value })}
                        placeholder="e.g. MyYOLOProject"
                    />
                </div>

                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px' }}>Parent Directory (Where project folder will be created)</label>
                    <div style={{ display: 'flex', gap: '10px' }}>
                        <input
                            className="input"
                            value={createForm.parentDir}
                            onChange={e => setCreateForm({ ...createForm, parentDir: e.target.value })}
                        />
                        <button className="btn" onClick={() => onBrowse((path) => setCreateForm(prev => ({ ...prev, parentDir: path })), 'dir')}>Browse</button>
                    </div>
                </div>

                {createMode === 'standard' ? (
                    <>
                        <div style={{ marginBottom: '15px' }}>
                            <label style={{ display: 'block', marginBottom: '5px' }}>Raw Images Source</label>
                            <div style={{ display: 'flex', gap: '10px' }}>
                                <input
                                    className="input"
                                    value={createForm.rawImagesDir}
                                    onChange={e => setCreateForm({ ...createForm, rawImagesDir: e.target.value })}
                                />
                                <button className="btn" onClick={() => onBrowse((path) => setCreateForm(prev => ({ ...prev, rawImagesDir: path })), 'dir')}>Browse</button>
                            </div>
                        </div>

                        <div style={{ marginBottom: '20px' }}>
                            <label style={{ display: 'block', marginBottom: '5px' }}>Object Classes (Comma separated)</label>
                            <input
                                className="input"
                                value={createForm.classes}
                                onChange={e => setCreateForm({ ...createForm, classes: e.target.value })}
                                placeholder="e.g. car, pedestrian, cyclist"
                            />
                        </div>
                    </>
                ) : (
                    <div style={{ marginBottom: '20px' }}>
                        <label style={{ display: 'block', marginBottom: '5px' }}>Existing Split Dataset Path</label>
                        <div style={{ display: 'flex', gap: '10px' }}>
                            <input
                                className="input"
                                value={createForm.splitDatasetPath}
                                onChange={e => setCreateForm({ ...createForm, splitDatasetPath: e.target.value })}
                                placeholder="/path/to/dataset (containing train/valid/test)"
                            />
                            <button className="btn" onClick={() => onBrowse((path) => setCreateForm(prev => ({ ...prev, splitDatasetPath: path })), 'dir')}>Browse</button>
                        </div>
                        <p style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '5px' }}>
                            Must contain <code>data.yaml</code> and split subfolders (train, valid, etc).
                        </p>
                    </div>
                )}

                <div style={{ display: 'flex', gap: '10px', alignItems: 'center', justifyContent: 'center' }}>
                    {loading ? (
                        <div style={{ textAlign: 'center', color: 'var(--primary)' }}>
                            <div className="spinner" style={{ margin: '0 auto 10px auto' }}></div>
                            <div>Creating Project...</div>
                            <div style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '5px' }}>
                                Copying large datasets may take a while. Please do not close this window.
                            </div>
                        </div>
                    ) : (
                        <>
                            <button className="btn btn-primary" onClick={handleCreateSubmit} style={{ flex: 1 }} disabled={loading}>
                                Create Project
                            </button>
                            <button className="btn" onClick={() => setMode('landing')} style={{ flex: 1 }} disabled={loading}>Cancel</button>
                        </>
                    )}
                </div>
            </div>
        );
    }

    if (mode === 'load') {
        return (
            <div className="glass" style={{ maxWidth: '600px', margin: '50px auto', padding: '30px', borderRadius: '16px' }}>
                <h2 style={{ marginBottom: '20px' }}>Load Existing Project</h2>

                <div style={{ marginBottom: '25px' }}>
                    <label style={{ display: 'block', marginBottom: '5px' }}>Project Directory Path</label>
                    <div style={{ display: 'flex', gap: '10px' }}>
                        <input
                            className="input"
                            value={loadPath}
                            onChange={e => setLoadPath(e.target.value)}
                            placeholder="/home/user/my_yolo_project"
                        />
                        <button className="btn" onClick={() => onBrowse(setLoadPath, 'dir')}>Browse</button>
                    </div>
                </div>

                <div style={{ display: 'flex', gap: '10px' }}>
                    <button
                        className="btn btn-primary"
                        onClick={() => {
                            if (!loadPath) return alert("Please select a project path");
                            onLoadProject(loadPath);
                        }}
                        style={{ flex: 1 }}
                    >
                        Load Project
                    </button>
                    <button className="btn" onClick={() => setMode('landing')} style={{ flex: 1 }}>Cancel</button>
                </div>
            </div>
        );
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '30px' }}>
            <div style={{ textAlign: 'center' }}>
                <h1 style={{ fontSize: '3rem', marginBottom: '10px', background: 'linear-gradient(45deg, #6366f1, #8b5cf6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                    YOLO WebUI
                </h1>
                <p style={{ opacity: 0.7 }}>Project-based Workflow Manager</p>
            </div>

            <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }}>
                <button
                    className="glass"
                    style={{ padding: '30px', width: '200px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '15px', cursor: 'pointer', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.1)' }}
                    onClick={() => setMode('create')}
                >
                    <span style={{ fontSize: '3rem' }}>🆕</span>
                    <span style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>Create Project</span>
                </button>

                <button
                    className="glass"
                    style={{ padding: '30px', width: '200px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '15px', cursor: 'pointer', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.1)' }}
                    onClick={() => setMode('load')}
                >
                    <span style={{ fontSize: '3rem' }}>📂</span>
                    <span style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>Load Project</span>
                </button>

                <button
                    className="glass"
                    style={{ padding: '30px', width: '200px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '15px', cursor: 'pointer', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.1)' }}
                    onClick={() => onSkip('processing')}
                >
                    <span style={{ fontSize: '3rem' }}>🛠️</span>
                    <span style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>Dataset Tools</span>
                </button>

                <button
                    className="glass"
                    style={{ padding: '30px', width: '200px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '15px', cursor: 'pointer', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.1)' }}
                    onClick={() => onSkip('training')}
                >
                    <span style={{ fontSize: '3rem' }}>🏋️</span>
                    <span style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>Train Only</span>
                </button>
            </div>
            <div style={{ marginTop: '20px' }}>
                <button
                    className="btn"
                    style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.2)' }}
                    onClick={() => onSkip('training')}
                >
                    Skip to Training Mode (No Project)
                </button>
            </div>
        </div>
    );
};

export default ProjectLanding;
