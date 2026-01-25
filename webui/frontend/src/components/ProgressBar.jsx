import React from 'react';

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

export default ProgressBar;
