import React, { useState } from 'react';

const ProjectLanding = ({ onCreateProject, onLoadProject, onBrowse }) => {
    const [mode, setMode] = useState('landing'); // landing, create, load
    const [loadPath, setLoadPath] = useState('');
    const [createForm, setCreateForm] = useState({
        name: '',
        parentDir: '',
        rawImagesDir: '',
        classes: '' // comma separated
    });

    const handleCreateSubmit = () => {
        if (!createForm.name || !createForm.parentDir || !createForm.rawImagesDir) {
            alert("Please fill in all required fields");
            return;
        }
        const classesList = createForm.classes.split(',').map(c => c.trim()).filter(c => c);
        onCreateProject({
            name: createForm.name,
            parent_dir: createForm.parentDir,
            raw_images_dir: createForm.rawImagesDir,
            classes: classesList
        });
    };

    if (mode === 'create') {
        return (
            <div className="glass" style={{ maxWidth: '600px', margin: '50px auto', padding: '30px', borderRadius: '16px' }}>
                <h2 style={{ marginBottom: '20px' }}>Create New Project</h2>

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

                <div style={{ display: 'flex', gap: '10px' }}>
                    <button className="btn btn-primary" onClick={handleCreateSubmit} style={{ flex: 1 }}>Create Project</button>
                    <button className="btn" onClick={() => setMode('landing')} style={{ flex: 1 }}>Cancel</button>
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

            <div style={{ display: 'flex', gap: '20px' }}>
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
            </div>
        </div>
    );
};

export default ProjectLanding;
