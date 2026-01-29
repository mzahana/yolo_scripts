import React from 'react';
import ProgressBar from '../ProgressBar';

const SplitDatasetTool = ({
    splitInput,
    setSplitInput,
    splitCount,
    setSplitCount,
    handleSplit,
    isTaskRunning,
    taskProgress,
    openFileBrowser
}) => {
    return (
        <div className="glass section-card">
            <div className="section-title">Split: Divide a dataset into splits</div>
            <p style={{ opacity: 0.7, marginBottom: '25px', fontSize: '0.9rem' }}>
                Split a dataset into multiple equal parts (in the same directory you enter).
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
    );
};

export default SplitDatasetTool;
