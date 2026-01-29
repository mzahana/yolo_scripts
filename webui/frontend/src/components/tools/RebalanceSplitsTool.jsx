import React from 'react';

const RebalanceSplitsTool = ({
    rebalanceInput,
    setRebalanceInput,
    rebalanceStats,
    newRebalancePcts,
    setNewRebalancePcts,
    deleteOriginalRebalance,
    setDeleteOriginalRebalance,
    handleRebalance,
    fetchRebalanceStats,
    isTaskRunning,
    openFileBrowser
}) => {
    return (
        <div className="glass section-card">
            <div className="section-title">Dataset Splits: Re-balance</div>
            <p style={{ opacity: 0.7, marginBottom: '25px', fontSize: '0.9rem' }}>
                Re-distribute images between Train, Validation, and Test sets.
            </p>

            <div className="input-group">
                <label>Dataset to Re-balance</label>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <input
                        type="text"
                        placeholder="/path/to/dataset"
                        value={rebalanceInput}
                        onChange={(e) => setRebalanceInput(e.target.value)}
                        onBlur={() => { if (rebalanceInput) fetchRebalanceStats(rebalanceInput); }}
                        style={{ flex: 1 }}
                    />
                    <button className="browse-btn" onClick={() => openFileBrowser('rebalance_input', 'dir')}>
                        Browse
                    </button>
                </div>
            </div>

            {rebalanceStats && (
                <div className="glass p-card" style={{ marginTop: '20px', padding: '15px', background: 'rgba(255,255,255,0.03)' }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '10px', fontSize: '0.9rem' }}>Current Splits:</div>
                    <div style={{ display: 'flex', gap: '20px', fontSize: '0.85rem' }}>
                        <span>Total: {rebalanceStats.total} images</span>
                        {Object.entries(rebalanceStats.splits).map(([k, v]) => (
                            <span key={k} style={{ color: 'var(--text-muted)' }}>
                                {k}: {v.count} ({v.percentage}%)
                            </span>
                        ))}
                    </div>
                </div>
            )}

            <div className="input-group" style={{ marginTop: '20px' }}>
                <label>New Split Ratios (%)</label>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
                    <div>
                        <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>Train</span>
                        <input
                            type="number"
                            value={newRebalancePcts.train}
                            onChange={(e) => setNewRebalancePcts({ ...newRebalancePcts, train: parseInt(e.target.value) || 0 })}
                        />
                    </div>
                    <div>
                        <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>Val</span>
                        <input
                            type="number"
                            value={newRebalancePcts.val}
                            onChange={(e) => setNewRebalancePcts({ ...newRebalancePcts, val: parseInt(e.target.value) || 0 })}
                        />
                    </div>
                    <div>
                        <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>Test</span>
                        <input
                            type="number"
                            value={newRebalancePcts.test}
                            onChange={(e) => setNewRebalancePcts({ ...newRebalancePcts, test: parseInt(e.target.value) || 0 })}
                        />
                    </div>
                </div>

                {(newRebalancePcts.train + newRebalancePcts.val + newRebalancePcts.test) !== 100 && (
                    <div style={{ color: '#ef4444', fontSize: '0.8rem', marginTop: '5px' }}>
                        Total: {newRebalancePcts.train + newRebalancePcts.val + newRebalancePcts.test}% (Must be 100%)
                    </div>
                )}
            </div>

            <div className="input-group" style={{ marginTop: '20px' }}>
                <label className="checkbox-container">
                    <input
                        type="checkbox"
                        checked={deleteOriginalRebalance}
                        onChange={(e) => setDeleteOriginalRebalance(e.target.checked)}
                    />
                    <span className="checkmark"></span>
                    <span style={{ marginLeft: '10px' }}>Delete Original Dataset (Replace in-place)</span>
                </label>
                <div style={{ fontSize: '0.8rem', opacity: 0.6, marginTop: '5px' }}>
                    {deleteOriginalRebalance
                        ? "Warning: The original folder will be replaced by the re-balanced version."
                        : "A new folder will be created (e.g., dataset_rebalanced)."}
                </div>
            </div>

            <button
                className="btn btn-primary"
                style={{ marginTop: '20px' }}
                onClick={handleRebalance}
                disabled={isTaskRunning || (newRebalancePcts.train + newRebalancePcts.val + newRebalancePcts.test) !== 100}
            >
                Re-balance Splits
            </button>
        </div>
    );
};

export default RebalanceSplitsTool;
