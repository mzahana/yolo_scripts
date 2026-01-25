import React, { useState } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    PieChart, Pie, Cell
} from 'recharts';
import ProgressBar from './ProgressBar';

const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'];

const StatsView = ({
    projectPaths,
    existingDatasets,
    datasetStats,
    statsPath,
    setStatsPath,
    fetchStats,
    handleExtractEmpty,
    isTaskRunning,
    taskProgress,
    datasetPath,
    projectConfig,
    activeTab,
    labelResult
}) => {
    // Prepare options
    const options = [
        { label: 'Entire Project (Current State)', value: projectPaths?.annotations || '' },
        ...existingDatasets.map(ds => ({ label: `Dataset: ${ds.name}`, value: ds.path }))
    ];

    const [statsScope, setStatsScope] = useState('combined');

    // Handle selection
    const handleSourceChange = (e) => {
        const val = e.target.value;
        if (val) {
            setStatsPath(val);
            fetchStats(val);
            setStatsScope('combined'); // Reset scope
        }
    };

    const getDisplayStats = () => {
        if (!datasetStats) return null;
        if (statsScope === 'combined' || !datasetStats.per_split_stats || !datasetStats.per_split_stats[statsScope]) {
            return datasetStats;
        }
        const s = datasetStats.per_split_stats[statsScope];
        return {
            total_images: s.total_images,
            total_objects: s.total_objects,
            empty_count: s.empty_images_count,
            class_stats: s.class_counts.map(c => ({ name: c.class, count: c.count, percentage: c.percentage })),
            empty_images: s.empty_images
        };
    };

    const displayStats = getDisplayStats();

    return (
        <div className="stats-container">
            <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <label style={{ fontWeight: 'bold' }}>Stats Source:</label>
                <select
                    className="input"
                    style={{ maxWidth: '300px' }}
                    value={statsPath || (projectPaths?.annotations || '')}
                    onChange={handleSourceChange}
                >
                    {options.map((opt, i) => (
                        <option key={i} value={opt.value}>{opt.label}</option>
                    ))}
                </select>
                <button className="btn btn-secondary" onClick={() => fetchStats(statsPath)}>🔄 Refresh</button>
            </div>

            {!displayStats ? (
                <div style={{ padding: '40px', textAlign: 'center', opacity: 0.5 }}>
                    Select a source to view statistics.
                </div>
            ) : (
                <>
                    {datasetStats.per_split_stats && (
                        <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
                            <button
                                className={`btn ${statsScope === 'combined' ? 'btn-primary' : 'btn-secondary'}`}
                                onClick={() => setStatsScope('combined')}
                            >
                                Combined
                            </button>
                            {Object.keys(datasetStats.per_split_stats).map(split => (
                                <button
                                    key={split}
                                    className={`btn ${statsScope === split ? 'btn-primary' : 'btn-secondary'}`}
                                    onClick={() => setStatsScope(split)}
                                    style={{ textTransform: 'capitalize' }}
                                >
                                    {split}
                                </button>
                            ))}
                        </div>
                    )}

                    <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', marginBottom: '30px' }}>
                        <div className="stats-card glass">
                            <div style={{ opacity: 0.6, fontSize: '0.9rem' }}>Total Images</div>
                            <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{displayStats.total_images}</div>
                        </div>
                        <div className="stats-card glass">
                            <div style={{ opacity: 0.6, fontSize: '0.9rem' }}>Total Objects</div>
                            <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{displayStats.total_objects}</div>
                        </div>
                        <div className="stats-card glass">
                            <div style={{ opacity: 0.6, fontSize: '0.9rem' }}>Empty Images</div>
                            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: displayStats.empty_count > 0 ? '#ef4444' : 'var(--accent)' }}>
                                {displayStats.empty_count}
                            </div>
                        </div>
                    </div>

                    <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '30px' }}>
                        <div className="glass section-card">
                            <div className="section-title" style={{ fontSize: '1.2rem', marginBottom: '20px' }}>Objects per Class</div>
                            <div style={{ height: '350px' }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={displayStats.class_stats} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                        <XAxis dataKey="name" stroke="var(--text-muted)" angle={-45} textAnchor="end" height={80} interval={0} />
                                        <YAxis stroke="var(--text-muted)" />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid var(--border-color)', borderRadius: '8px' }}
                                            itemStyle={{ color: 'white' }}
                                        />
                                        <Bar dataKey="count" fill="var(--primary)" radius={[4, 4, 0, 0]}>
                                            {datasetStats.class_stats.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        <div className="glass section-card">
                            <div className="section-title" style={{ fontSize: '1.2rem', marginBottom: '20px' }}>Class Distribution (%)</div>
                            <div style={{ height: '350px' }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={displayStats.class_stats}
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={60}
                                            outerRadius={100}
                                            fill="#8884d8"
                                            paddingAngle={5}
                                            dataKey="count"
                                            nameKey="name"
                                        >
                                            {displayStats.class_stats.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                            ))}
                                        </Pie>
                                        <Tooltip
                                            contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid var(--border-color)', borderRadius: '8px' }}
                                            itemStyle={{ color: 'white' }}
                                        />
                                        <Legend />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>

                    <div className="glass section-card">
                        <div className="section-title" style={{ fontSize: '1.2rem', marginBottom: '20px' }}>Detailed Report</div>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                            <thead>
                                <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border-color)' }}>
                                    <th style={{ padding: '15px' }}>Class Name</th>
                                    <th style={{ padding: '15px' }}>Count</th>
                                    <th style={{ padding: '15px' }}>Percentage</th>
                                </tr>
                            </thead>
                            <tbody>
                                {displayStats.class_stats.map((stat, i) => (
                                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                        <td style={{ padding: '15px', fontWeight: 'bold' }}>{stat.name}</td>
                                        <td style={{ padding: '15px' }}>{stat.count}</td>
                                        <td style={{ padding: '15px' }}>{stat.percentage}%</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    <div className="glass section-card" style={{ marginTop: '30px' }}>
                        <div className="section-title" style={{ fontSize: '1.2rem', color: '#ef4444', marginBottom: '15px' }}>Images with No Detections ({displayStats.empty_count})</div>
                        {displayStats.empty_count > 0 ? (
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '10px', marginTop: '10px' }}>
                                {displayStats.empty_images.slice(0, 12).map((imgName, i) => (
                                    <div key={i} className="card" style={{ padding: '10px', fontSize: '0.8rem', textAlign: 'center' }}>
                                        {imgName}
                                    </div>
                                ))}
                                {displayStats.empty_images.length > 12 && (
                                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.5, fontSize: '0.8rem' }}>
                                        +{displayStats.empty_images.length - 12} more
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div style={{ opacity: 0.5, fontStyle: 'italic', padding: '20px' }}>
                                All images have detections!
                            </div>
                        )}

                        {displayStats.empty_count > 0 && (
                            <div style={{ marginTop: '20px' }}>
                                <div style={{ marginTop: '25px', display: 'flex', flexDirection: 'column', gap: '15px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                                        <button
                                            className="btn btn-secondary"
                                            style={{ borderColor: '#ef4444', color: '#ef4444' }}
                                            onClick={() => handleExtractEmpty(displayStats.empty_images)}
                                            disabled={isTaskRunning}
                                        >
                                            📦 Extract Empty Images to a Folder
                                        </button>
                                    </div>
                                    <ProgressBar progress={taskProgress} type="extracting_empty" />
                                </div>
                            </div>
                        )}
                    </div>
                </>
            )}
        </div>
    );
};

export default StatsView;
