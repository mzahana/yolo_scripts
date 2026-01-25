import React from 'react';

const LightboxModal = ({ selectedLightboxImage, onClose, maskedMountUrl, cacheBuster }) => {
    if (!selectedLightboxImage) return null;
    return (
        <div className="modal-overlay" onClick={onClose} style={{ background: 'rgba(0,0,0,0.9)', zIndex: 2000 }}>
            <div className="lightbox-content" onClick={e => e.stopPropagation()} style={{ position: 'relative', maxWidth: '90vw', maxHeight: '90vh' }}>
                <img
                    src={`http://localhost:8000${maskedMountUrl}/${selectedLightboxImage.name}?t=${cacheBuster}`}
                    style={{ width: '100%', height: 'auto', borderRadius: '12px', boxShadow: '0 0 40px rgba(0,0,0,0.5)' }}
                    alt="Enlarged"
                />
                <button
                    className="browse-btn"
                    style={{ position: 'absolute', top: '20px', right: '20px', background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(255,255,255,0.2)' }}
                    onClick={onClose}
                >
                    ✕ Close
                </button>
                <div style={{ padding: '20px 0', color: 'white' }}>
                    <h3 style={{ margin: 0 }}>{selectedLightboxImage.name}</h3>
                    <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                        {Object.entries(selectedLightboxImage.stats).map(([cls, count]) => (
                            <span key={cls} className="badge" style={{ fontSize: '0.9rem' }}>{cls}: {count}</span>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LightboxModal;
