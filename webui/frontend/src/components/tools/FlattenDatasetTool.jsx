import React from 'react';

const FlattenDatasetTool = ({
    flattenInput,
    setFlattenInput,
    handleFlatten,
    isTaskRunning,
    openFileBrowser
}) => {
    return (
        <div className="glass section-card">
            <div className="section-title">Dataset Tools: Flatten</div>
            <p style={{ opacity: 0.7, marginBottom: '25px', fontSize: '0.9rem' }}>
                Convert a split dataset (train/val/test) into a flat structure (images/labels).
            </p>

            <div className="input-group">
                <label>Dataset to Flatten</label>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <input
                        type="text"
                        placeholder="/path/to/dataset"
                        value={flattenInput}
                        onChange={(e) => setFlattenInput(e.target.value)}
                        style={{ flex: 1 }}
                    />
                    <button className="browse-btn" onClick={() => openFileBrowser('flatten_input', 'dir')}>
                        Browse
                    </button>
                </div>
            </div>

            <button
                className="btn btn-primary"
                onClick={handleFlatten}
                disabled={isTaskRunning || !flattenInput}
                style={{ marginTop: '20px' }}
            >
                Flatten Dataset
            </button>
        </div>
    );
};

export default FlattenDatasetTool;
