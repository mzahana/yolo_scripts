import React from 'react';
import ProgressBar from '../ProgressBar';

const MergeDatasetsTool = ({
    mergeSources,
    setMergeSources,
    addMergeSource,
    removeMergeSource,
    mergeOutput,
    setMergeOutput,
    handleMerge,
    isTaskRunning,
    taskProgress,
    openFileBrowser
}) => {
    return (
        <div className="glass section-card">
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
        </div>
    );
};

export default MergeDatasetsTool;
