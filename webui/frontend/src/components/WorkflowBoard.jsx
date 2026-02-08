
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = '/api';

const WorkflowBoard = ({ projectPath, currentUser, onOpenJob, onOpenReview }) => {
    const [unassignedCount, setUnassignedCount] = useState(0);
    const [datasetCount, setDatasetCount] = useState(0);
    const [jobs, setJobs] = useState([]);
    const [loading, setLoading] = useState(false);
    const [lastSync, setLastSync] = useState(new Date());
    const [isAutoRefresh, setIsAutoRefresh] = useState(true);

    // Modal State
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [createJobSource, setCreateJobSource] = useState('unassigned'); // unassigned or dataset
    const [batchSize, setBatchSize] = useState(50);
    const [assignee, setAssignee] = useState(currentUser || '');
    const [reviewer, setReviewer] = useState('');

    useEffect(() => {
        if (projectPath) {
            fetchData();
        }
    }, [projectPath]);

    // Polling logic
    useEffect(() => {
        if (!projectPath || !isAutoRefresh) return;

        const interval = setInterval(() => {
            // Only refresh if tab is visible to save resources
            if (document.visibilityState === 'visible') {
                fetchData();
            }
        }, 5000);

        return () => clearInterval(interval);
    }, [projectPath, isAutoRefresh]);

    const fetchData = async () => {
        // Only show full-screen loader on first load to avoid flicker during polling
        if (jobs.length === 0) setLoading(true);

        const ts = new Date().getTime();

        try {
            // Fetch independently so one failure doesn't break the others
            const p1 = axios.get(`${API_BASE}/workflow/unassigned?project_path=${encodeURIComponent(projectPath)}&t=${ts}`)
                .then(res => setUnassignedCount(res.data.count))
                .catch(err => console.error("Error fetching unassigned count:", err));

            const p2 = axios.get(`${API_BASE}/workflow/jobs?project_path=${encodeURIComponent(projectPath)}&t=${ts}`)
                .then(res => setJobs(res.data))
                .catch(err => console.error("Error fetching jobs:", err));

            const p3 = axios.get(`${API_BASE}/workflow/dataset?project_path=${encodeURIComponent(projectPath)}&t=${ts}`)
                .then(res => {
                    setDatasetCount(res.data.count);
                    console.log("DEBUG: Dataset count updated to:", res.data.count);
                })
                .catch(err => console.error("Error fetching dataset count:", err));

            await Promise.all([p1, p2, p3]);
            setLastSync(new Date());
        } finally {
            setLoading(false);
        }
    };

    const [includeAnnotated, setIncludeAnnotated] = useState(false);
    const [selectedSourceJob, setSelectedSourceJob] = useState('');

    const openCreateModal = (source) => {
        setCreateJobSource(source);
        setAssignee(currentUser || '');
        setReviewer('');
        setIncludeAnnotated(false);
        setSelectedSourceJob('');
        setShowCreateModal(true);
    };

    const handleCreateJob = async () => {
        try {
            await axios.post(`${API_BASE}/workflow/job/create`, {
                project_path: projectPath,
                source: createJobSource,
                batch_size: parseInt(batchSize) || 100,
                annotator: assignee,
                reviewer: reviewer,
                include_annotated: includeAnnotated,
                source_job_id: selectedSourceJob || null
            });
            setShowCreateModal(false);
            fetchData();
        } catch (err) {
            alert("Error creating job: " + (err.response?.data?.detail || err.message));
        }
    };

    const handleUnassignJob = async (job) => {
        if (!confirm(`Are you sure you want to unassign "${job.name}"? Images will return to the unassigned pool.`)) return;
        try {
            await axios.post(`${API_BASE}/workflow/job/${job.id}/unassign`, {
                project_path: projectPath
            });
            fetchData();
        } catch (err) {
            alert("Error unassigning job: " + (err.response?.data?.detail || err.message));
        }
    };

    const handleResetDataset = async () => {
        if (!confirm(`Are you sure you want to RESET the dataset? This will DELETE ALL ANNOTATIONS in the project annotation folder and move all images back to 'Unassigned'. This cannot be undone.`)) return;
        try {
            await axios.post(`${API_BASE}/workflow/dataset/reset`, {
                project_path: projectPath
            });
            fetchData();
        } catch (err) {
            alert("Error resetting dataset: " + (err.response?.data?.detail || err.message));
        }
    };

    const handleSubmitJob = async (job) => {
        if (!confirm(`Submit job "${job.name}" for review?`)) return;
        try {
            await axios.post(`${API_BASE}/workflow/job/${job.id}/submit`, {
                project_path: projectPath
            });
            fetchData();
        } catch (err) {
            alert("Error submitting job: " + (err.response?.data?.detail || err.message));
        }
    };

    const handleApproveJob = async (job) => {
        if (!confirm(`Approve all remaining images in "${job.name}" and mark job as done?`)) return;
        try {
            await axios.post(`${API_BASE}/workflow/job/${job.id}/approve_all`, {
                project_path: projectPath
            });
            fetchData();
        } catch (err) {
            alert("Error approving job: " + (err.response?.data?.detail || err.message));
        }
    };

    // Columns
    const annotatingJobs = jobs.filter(j => j.status === 'annotating');
    const reviewJobs = jobs.filter(j => j.status === 'review');
    const doneJobs = jobs.filter(j => j.status === 'done');

    const isAdmin = currentUser === 'admin';

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            {/* Sync Status Bar */}
            <div style={{
                display: 'flex',
                justifyContent: 'flex-end',
                alignItems: 'center',
                gap: '15px',
                padding: '10px 20px',
                background: 'rgba(255,255,255,0.02)',
                borderBottom: '1px solid var(--border-color)',
                fontSize: '0.8rem',
                color: 'var(--text-muted)'
            }}>
                <div style={{ marginRight: 'auto', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-bright)' }}>
                    <span>Active User:</span>
                    <strong style={{
                        background: 'rgba(99,102,241,0.1)',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        color: isAdmin ? '#f59e0b' : '#6366f1',
                        border: `1px solid ${isAdmin ? '#f59e0b' : '#6366f1'}`
                    }}>
                        {currentUser || 'None'} {isAdmin && '(Admin Mode)'}
                    </strong>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <div style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: isAutoRefresh ? '#10b981' : '#6b7280',
                        boxShadow: isAutoRefresh ? '0 0 8px #10b981' : 'none'
                    }}></div>
                    <span>{isAutoRefresh ? 'Live Sync Active' : 'Auto-Refresh Paused'}</span>
                </div>
                <span>Last updated: {lastSync.toLocaleTimeString()}</span>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                        className="btn"
                        style={{ padding: '2px 10px', fontSize: '0.7rem', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)' }}
                        onClick={() => setIsAutoRefresh(!isAutoRefresh)}
                    >
                        {isAutoRefresh ? 'Pause' : 'Resume'}
                    </button>
                    <button
                        className="btn"
                        style={{ padding: '2px 10px', fontSize: '0.7rem', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)' }}
                        onClick={fetchData}
                        disabled={loading}
                    >
                        Refresh Now
                    </button>
                </div>
            </div>

            <div className="workflow-board" style={{ display: 'flex', gap: '20px', padding: '20px', flex: 1, overflowX: 'auto' }}>

                {/* Unassigned Column */}
                <div className="board-column">
                    <h3>Unassigned</h3>
                    <div className="card">
                        <div className="stat-number">{unassignedCount}</div>
                        <div>Images Available</div>
                        <button className="primary-btn" style={{ marginTop: '10px', width: '100%' }} onClick={() => openCreateModal('unassigned')}>
                            Create Annotation Job
                        </button>
                        <div style={{ fontSize: '0.8em', color: '#888', marginTop: '5px' }}>
                            Creates a batch for manual annotation.
                        </div>
                    </div>
                </div>

                {/* Annotating Column */}
                <div className="board-column">
                    <h3>Annotating</h3>
                    <div className="job-list">
                        {annotatingJobs.map(job => (
                            <div key={job.id} className="job-card">
                                <div className="job-header">
                                    <strong>{job.name}</strong>
                                    <span className={`status-tag ${job.status}`}>{job.status}</span>
                                </div>
                                <div className="job-details">
                                    <div>Assignee: {job.annotator}</div>

                                    {/* Progress Bar */}
                                    <div style={{
                                        height: '8px',
                                        backgroundColor: '#eee',
                                        borderRadius: '4px',
                                        margin: '5px 0',
                                        overflow: 'hidden'
                                    }}>
                                        <div style={{
                                            width: `${(job.stats.done / (job.stats.total || 1)) * 100}%`,
                                            height: '100%',
                                            backgroundColor: '#4CAF50',
                                            transition: 'width 0.3s'
                                        }}></div>
                                    </div>

                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9em', color: '#666' }}>
                                        <span>Labeled: {job.stats.done}</span>
                                        <span>Unlabeled: {job.stats.pending + (job.stats.rejected || 0)}</span>
                                    </div>

                                    {job.stats.rejected > 0 && <div style={{ color: 'red', fontSize: '0.85em', marginTop: '2px' }}>Rejected: {job.stats.rejected} (Needs Fix)</div>}
                                </div>
                                <div style={{ display: 'flex', gap: '5px', marginTop: '10px' }}>
                                    <button className="action-btn" style={{ flex: 1 }} onClick={() => onOpenJob(job)}>
                                        {(currentUser === job.annotator || isAdmin) ? "Continue" : "View"}
                                    </button>
                                    {(currentUser === job.annotator || isAdmin) && (
                                        <button
                                            className="primary-btn"
                                            style={{ flex: 1, fontSize: '0.9em' }}
                                            onClick={(e) => { e.stopPropagation(); handleSubmitJob(job); }}
                                        >
                                            Submit
                                        </button>
                                    )}
                                </div>
                                <button
                                    className="secondary-btn"
                                    style={{ marginTop: '5px', width: '100%', fontSize: '0.8em', opacity: 0.7 }}
                                    onClick={(e) => { e.stopPropagation(); handleUnassignJob(job); }}
                                >
                                    Unassign (Job)
                                </button>
                            </div>
                        ))}
                        {annotatingJobs.length === 0 && <div className="empty-state">No active annotation jobs</div>}
                    </div>
                </div>

                {/* Review Column */}
                <div className="board-column">
                    <h3>Review</h3>
                    <div className="job-list">
                        {reviewJobs.map(job => (
                            <div key={job.id} className="job-card">
                                <div className="job-header">
                                    <strong>{job.name}</strong>
                                    <span className={`status-tag ${job.status}`}>{job.status}</span>
                                </div>
                                <div className="job-details">
                                    <div>Reviewer: {job.reviewer}</div>
                                    <div>Pending: {job.stats.pending}</div>
                                </div>
                                <button className="action-btn" onClick={() => onOpenReview(job)}>
                                    {(currentUser === job.reviewer || !job.reviewer || isAdmin) ? "Start Review" : "View Status"}
                                </button>
                                {(currentUser === job.reviewer || !job.reviewer || isAdmin) && (
                                    <button
                                        className="primary-btn"
                                        style={{ marginTop: '5px', width: '100%', fontSize: '0.9em', background: '#10b981' }}
                                        onClick={(e) => { e.stopPropagation(); handleApproveJob(job); }}
                                    >
                                        Approve Job
                                    </button>
                                )}
                            </div>
                        ))}
                        {reviewJobs.length === 0 && <div className="empty-state">No jobs pending review</div>}
                    </div>
                </div>

                {/* Dataset Column */}
                <div className="board-column">
                    <h3>Dataset</h3>
                    <div className="card">
                        <div className="stat-number">{datasetCount}</div>
                        <div>Images in Project Dataset</div>
                        {doneJobs.length === 0 && datasetCount > 0 && (
                            <button
                                className="danger-btn"
                                style={{ marginTop: '10px', width: '100%', fontSize: '0.9em' }}
                                onClick={() => handleResetDataset()}
                            >
                                Reset Dataset to Unassigned
                            </button>
                        )}
                        <div style={{ fontSize: '0.8em', color: '#888', marginTop: '5px' }}>
                            {doneJobs.length > 0 ? "Completed jobs." : "Images ready for export or reset."}
                        </div>
                    </div>

                    <h4 style={{ marginTop: '20px' }}>Completed Jobs</h4>
                    <div className="job-list" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                        {doneJobs.map(job => (
                            <div key={job.id} className="job-card done">
                                <div className="job-header">
                                    <strong>{job.name}</strong>
                                </div>
                                <div className="job-details">
                                    <div>{job.stats.done} Images</div>
                                    <button
                                        className="secondary-btn"
                                        style={{ marginTop: '5px', width: '100%', fontSize: '0.8em' }}
                                        onClick={() => handleUnassignJob(job)}
                                    >
                                        Unassign (Release)
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Create Job Modal */}
                {showCreateModal && (
                    <div className="modal-overlay">
                        <div className="modal-content glass" style={{ maxWidth: '400px' }}>
                            <h3>Create {createJobSource === 'dataset' ? 'Review' : 'Annotation'} Job</h3>

                            {createJobSource === 'unassigned' && (
                                <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: '10px' }}>
                                    <input
                                        type="checkbox"
                                        checked={includeAnnotated}
                                        onChange={e => setIncludeAnnotated(e.target.checked)}
                                    />
                                    <label style={{ marginBottom: 0 }}>Include images with existing annotations</label>
                                </div>
                            )}

                            {createJobSource === 'dataset' && (
                                <div className="form-group">
                                    <label>Source Scope</label>
                                    <select
                                        className="input"
                                        value={selectedSourceJob}
                                        onChange={e => setSelectedSourceJob(e.target.value)}
                                    >
                                        <option value="" disabled>Select a Job...</option>
                                        {doneJobs.map(j => (
                                            <option key={j.id} value={j.id}>{j.name} ({j.stats.done} imgs)</option>
                                        ))}
                                    </select>
                                </div>
                            )}

                            <div className="form-group">
                                <label>
                                    {createJobSource === 'dataset' ? 'Image Count (Ignored)' : 'Batch Size (Images)'}
                                </label>
                                <input
                                    type="number"
                                    value={batchSize}
                                    onChange={e => setBatchSize(e.target.value)}
                                    min="1"
                                    disabled={createJobSource === 'dataset'}
                                />
                            </div>

                            <div className="form-group">
                                <label>Annotator Name</label>
                                <input type="text" value={assignee} onChange={e => setAssignee(e.target.value)} placeholder="Who will annotate/fix?" />
                            </div>

                            <div className="form-group">
                                <label>Reviewer Name</label>
                                <input type="text" value={reviewer} onChange={e => setReviewer(e.target.value)} placeholder="Who will review?" />
                            </div>

                            <div className="modal-actions">
                                <button onClick={() => setShowCreateModal(false)}>Cancel</button>
                                <button className="primary-btn" onClick={handleCreateJob}>Create Job</button>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <style>{`
                .board-column {
                    flex: 1;
                    min-width: 250px;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                    padding: 10px;
                    display: flex;
                    flex-direction: column;
                }
                .board-column h3 {
                    margin-top: 0;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                    padding-bottom: 10px;
                    text-align: center;
                }
                .card {
                    background: rgba(30,30,30,0.6);
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }
                .stat-number {
                    font-size: 2em;
                    font-weight: bold;
                    color: #4ade80;
                }
                .job-list {
                    flex: 1;
                    overflow-y: auto;
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                .job-card {
                    background: rgba(50,50,50,0.6);
                    padding: 10px;
                    border-radius: 6px;
                    border: 1px solid rgba(255,255,255,0.1);
                }
                .job-card.done {
                    opacity: 0.7;
                }
                .job-header {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 5px;
                }
                .status-tag {
                    font-size: 0.8em;
                    padding: 2px 6px;
                    border-radius: 4px;
                    background: #333;
                }
                .status-tag.annotating { background: #3b82f6; }
                .status-tag.review { background: #f59e0b; }
                .status-tag.done { background: #10b981; }
                
                .job-details {
                    font-size: 0.9em;
                    color: #ccc;
                    margin-bottom: 10px;
                }
                .action-btn {
                    width: 100%;
                    padding: 5px;
                    background: rgba(255,255,255,0.1);
                    border: none;
                    color: white;
                    cursor: pointer;
                    border-radius: 4px;
                }
                .action-btn:hover {
                    background: rgba(255,255,255,0.2);
                }
                .form-group {
                    margin-bottom: 15px;
                }
                .form-group label {
                    display: block;
                    margin-bottom: 5px;
                }
                .form-group input {
                    width: 100%;
                    padding: 8px;
                    background: #333;
                    border: 1px solid #555;
                    color: white;
                    border-radius: 4px;
                }
                .primary-btn {
                    background: #3b82f6;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .secondary-btn {
                    background: #6366f1;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .modal-actions {
                    display: flex;
                    justify-content: flex-end;
                    gap: 10px;
                    margin-top: 20px;
                }
            `}</style>
        </div>
    );
};

export default WorkflowBoard;
