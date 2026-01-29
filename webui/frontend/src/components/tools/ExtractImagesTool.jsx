import React from 'react';
import ProgressBar from '../ProgressBar';

const ExtractImagesTool = ({
    extractSource,
    setExtractSource,
    availableClasses,
    selectedClasses,
    handleFetchClasses,
    toggleClass,
    toggleFilterClass,
    extractOutput,
    setExtractOutput,
    handleExtract,
    isTaskRunning,
    taskProgress,
    openFileBrowser
}) => {
    return (
        <div className="glass section-card">
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

            {availableClasses.length > 0 && (<>
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
                                <span onClick={(e) => { e.preventDefault(); toggleFilterClass(cls); }}>
                                    {cls}
                                </span>
                            </label>
                        ))}
                    </div>
                </div>
                <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'flex-end', gap: '15px' }}>
                    <button
                        className="btn btn-primary"
                        onClick={handleExtract}
                        disabled={isTaskRunning || selectedClasses.length === 0}
                    >
                        Extract Images
                    </button>
                </div>
            </>)}

            <div className="input-group" style={{ marginTop: '30px' }}>
                <label>Output Directory</label>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <input
                        type="text"
                        placeholder="/path/to/extracted/output"
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
    );
};

export default ExtractImagesTool;
