import React from 'react';
import ProgressBar from './ProgressBar';

const VerificationGallery = ({
    availableClasses,
    datasetStats,
    filterClasses = [],
    setFilterClasses,
    searchQuery,
    setSearchQuery,
    maskedOffset,
    maskedImages,
    totalMasked,
    isTaskRunning,
    handleGenerateMasks,
    handleRefreshGallery,
    toggleFilterClass,
    hasLabels,
    hasMasks,
    taskProgress,
    setSelectedLightboxImage,
    maskedMountUrl,
    cacheBuster,
    onJumpToAnnotation,
    handleFilterImage,
    MASKED_LIMIT,
    handlePageChange,
    verificationScroll,
    setVerificationScroll,
    dynamicRenderPath, // New prop for dynamic rendering source path
    selectedSplit,
    onSplitChange,
    datasetSplits = []
}) => {
    // Use labels classes if available
    const rawClasses = availableClasses.length > 0 ? availableClasses : (datasetStats?.class_stats?.map(s => s.name) || []);
    // Ensure "Empty" is an option
    const classes = rawClasses.includes("Empty") ? rawClasses : [...rawClasses, "Empty"];

    // Restore scroll position
    React.useLayoutEffect(() => {
        const scrollContainer = document.querySelector('.main-content'); // Assuming main-content is the scrollable area
        if (scrollContainer && verificationScroll > 0) {
            scrollContainer.scrollTop = verificationScroll;
        }

        // Save scroll on unmount/change
        return () => {
            if (scrollContainer) {
                setVerificationScroll(scrollContainer.scrollTop);
            }
        };
    }, []);

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

                    {/* Split Selector (if split dataset) */}
                    {datasetSplits.length > 0 && (
                        <div style={{ marginLeft: '10px' }}>
                            <select
                                className="input"
                                value={selectedSplit}
                                onChange={(e) => onSplitChange(e.target.value)}
                                style={{ width: '120px', padding: '8px 12px', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)', borderRadius: '8px', color: 'white' }}
                            >
                                {datasetSplits.map(s => (
                                    <option key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</option>
                                ))}
                            </select>
                        </div>
                    )}

                    {/* Search Input */}
                    <div style={{ flex: 1, margin: '0 20px', maxWidth: '300px' }}>
                        <div className="input-group">
                            <span className="input-prefix">🔎</span>
                            <input
                                type="text"
                                className="input"
                                placeholder="Search images..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                style={{ paddingLeft: '35px' }}
                            />
                            {searchQuery && (
                                <button
                                    onClick={() => setSearchQuery('')}
                                    style={{ position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', opacity: 0.5 }}
                                >
                                    ✕
                                </button>
                            )}
                        </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                        <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>
                            Showing {maskedOffset + 1}-{Math.min(maskedOffset + maskedImages.length, totalMasked)} of {totalMasked}
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
                            // Fix: toggleFilterClass was passed, but we need to check if it expects arg or not.
                            // App.jsx used toggleFilterClass(cls).
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
                                        src={maskedMountUrl ? `${maskedMountUrl}/${img.name}?t=${cacheBuster}` : (dynamicRenderPath ? `/api/dataset/render_image?path=${encodeURIComponent(dynamicRenderPath)}&name=${img.name}&t=${cacheBuster}` : '')}

                                        style={{ width: '100%', height: 'auto', display: 'block', transition: 'transform 0.3s' }}
                                        className="gallery-img"
                                        alt={img.name}
                                        loading="lazy"
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
                        totalMasked > MASKED_LIMIT && (
                            <div style={{ marginTop: '40px', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '20px' }}>
                                <button
                                    className="btn"
                                    disabled={maskedOffset === 0}
                                    onClick={() => handlePageChange(maskedOffset - MASKED_LIMIT)}
                                >
                                    ← Previous
                                </button>

                                <span style={{ opacity: 0.7 }}>
                                    Page {Math.floor(maskedOffset / MASKED_LIMIT) + 1} of {Math.ceil(totalMasked / MASKED_LIMIT)}
                                </span>

                                <button
                                    className="btn"
                                    disabled={maskedOffset + MASKED_LIMIT >= totalMasked}
                                    onClick={() => handlePageChange(maskedOffset + MASKED_LIMIT)}
                                >
                                    Next →
                                </button>
                            </div>
                        )
                    }
                </div>
            )}
        </div>
    );
};

export default VerificationGallery;
