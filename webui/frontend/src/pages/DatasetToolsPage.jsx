import React, { useState } from 'react';
import MergeDatasetsTool from '../components/tools/MergeDatasetsTool';
import ExtractImagesTool from '../components/tools/ExtractImagesTool';
import RebalanceSplitsTool from '../components/tools/RebalanceSplitsTool';
import FlattenDatasetTool from '../components/tools/FlattenDatasetTool';
import SplitDatasetTool from '../components/tools/SplitDatasetTool';
import DataAugmentationTool from '../components/tools/DataAugmentationTool';
import DatasetSampling from '../components/DatasetSampling';

const DatasetToolsPage = ({
    // Props for Merge
    mergeSources, setMergeSources, addMergeSource, removeMergeSource, mergeOutput, setMergeOutput, handleMerge,
    // Props for Extract
    extractSource, setExtractSource, availableClasses, selectedClasses, handleFetchClasses, toggleClass, toggleFilterClass, extractOutput, setExtractOutput, handleExtract,
    // Props for Rebalance
    rebalanceInput, setRebalanceInput, rebalanceStats, newRebalancePcts, setNewRebalancePcts, deleteOriginalRebalance, setDeleteOriginalRebalance, handleRebalance, fetchRebalanceStats,
    // Props for Flatten
    flattenInput, setFlattenInput, handleFlatten,
    // Props for Split
    splitInput, setSplitInput, splitCount, setSplitCount, handleSplit,
    // Props for Sampling
    datasetPath, projectConfig, handleLandingBrowse,
    // Common
    isTaskRunning, taskProgress, openFileBrowser, setIsTaskRunning, showNotification
}) => {
    const [activeTool, setActiveTool] = useState('merge');

    const tools = [
        { id: 'merge', label: 'Merge Datasets', icon: '🔄', description: 'Consolidate multiple labeled datasets into one.' },
        { id: 'extract', label: 'Extract Images', icon: '📦', description: 'Extract images by class from a labeled dataset.' },
        { id: 'rebalance', label: 'Re-balance Splits', icon: '⚖️', description: 'Re-distribute images between Train/Val/Test.' },
        { id: 'flatten', label: 'Flatten Dataset', icon: '📄', description: 'Convert split dataset to flat structure.' },
        { id: 'split', label: 'Split Dataset', icon: '✂️', description: 'Split a dataset into multiple parts.' },
        { id: 'augmentation', label: 'Data Augmentation', icon: '🎨', description: 'Augment dataset properties and objects.' },
        { id: 'sampling', label: 'Dataset Sampling', icon: '🎲', description: 'Sample a subset of images from a dataset.' },
    ];

    return (
        <div className="dataset-tools-page" style={{ display: 'flex', height: 'calc(100vh - 40px)', gap: '20px' }}>
            <aside className="tools-sidebar" style={{
                width: '280px',
                background: 'rgba(255, 255, 255, 0.03)',
                borderRadius: '12px',
                padding: '20px',
                display: 'flex',
                flexDirection: 'column',
                gap: '10px'
            }}>
                <div className="section-title" style={{ marginBottom: '10px' }}>Dataset Tools</div>
                {tools.map(tool => (
                    <div
                        key={tool.id}
                        onClick={() => setActiveTool(tool.id)}
                        style={{
                            padding: '12px 15px',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            background: activeTool === tool.id ? 'var(--primary)' : 'transparent',
                            color: activeTool === tool.id ? 'white' : 'var(--text-color)',
                            transition: 'all 0.2s',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '10px',
                            border: activeTool === tool.id ? 'none' : '1px solid transparent'
                        }}
                        onMouseEnter={(e) => {
                            if (activeTool !== tool.id) e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                        }}
                        onMouseLeave={(e) => {
                            if (activeTool !== tool.id) e.currentTarget.style.background = 'transparent';
                        }}
                    >
                        <span>{tool.icon}</span>
                        <div>
                            <div style={{ fontWeight: '500' }}>{tool.label}</div>
                            <div style={{ fontSize: '0.75rem', opacity: 0.7, lineHeight: '1.2' }}>{tool.description}</div>
                        </div>
                    </div>
                ))}
            </aside>

            <main className="tool-content" style={{ flex: 1, overflowY: 'auto' }}>
                {activeTool === 'merge' && (
                    <MergeDatasetsTool
                        mergeSources={mergeSources}
                        setMergeSources={setMergeSources}
                        addMergeSource={addMergeSource}
                        removeMergeSource={removeMergeSource}
                        mergeOutput={mergeOutput}
                        setMergeOutput={setMergeOutput}
                        handleMerge={handleMerge}
                        isTaskRunning={isTaskRunning}
                        taskProgress={taskProgress}
                        openFileBrowser={openFileBrowser}
                    />
                )}
                {activeTool === 'extract' && (
                    <ExtractImagesTool
                        extractSource={extractSource}
                        setExtractSource={setExtractSource}
                        availableClasses={availableClasses}
                        selectedClasses={selectedClasses}
                        handleFetchClasses={handleFetchClasses}
                        toggleClass={toggleClass}
                        toggleFilterClass={toggleFilterClass}
                        extractOutput={extractOutput}
                        setExtractOutput={setExtractOutput}
                        handleExtract={handleExtract}
                        isTaskRunning={isTaskRunning}
                        taskProgress={taskProgress}
                        openFileBrowser={openFileBrowser}
                    />
                )}
                {activeTool === 'rebalance' && (
                    <RebalanceSplitsTool
                        rebalanceInput={rebalanceInput}
                        setRebalanceInput={setRebalanceInput}
                        rebalanceStats={rebalanceStats}
                        newRebalancePcts={newRebalancePcts}
                        setNewRebalancePcts={setNewRebalancePcts}
                        deleteOriginalRebalance={deleteOriginalRebalance}
                        setDeleteOriginalRebalance={setDeleteOriginalRebalance}
                        handleRebalance={handleRebalance}
                        fetchRebalanceStats={fetchRebalanceStats}
                        isTaskRunning={isTaskRunning}
                        openFileBrowser={openFileBrowser}
                    />
                )}
                {activeTool === 'flatten' && (
                    <FlattenDatasetTool
                        flattenInput={flattenInput}
                        setFlattenInput={setFlattenInput}
                        handleFlatten={handleFlatten}
                        isTaskRunning={isTaskRunning}
                        openFileBrowser={openFileBrowser}
                    />
                )}
                {activeTool === 'split' && (
                    <SplitDatasetTool
                        splitInput={splitInput}
                        setSplitInput={setSplitInput}
                        splitCount={splitCount}
                        setSplitCount={setSplitCount}
                        handleSplit={handleSplit}
                        isTaskRunning={isTaskRunning}
                        taskProgress={taskProgress}
                        openFileBrowser={openFileBrowser}
                    />
                )}
                {activeTool === 'augmentation' && (
                    <DataAugmentationTool
                        datasetPath={datasetPath}
                        setDatasetPath={() => { }} // Read-only from context usually, or passing local setter if needed? The Prop name suggests passing generic setter, but usually datasetPath is from parent.
                        // Actually let's use the pattern from other tools if available or just internal state
                        // The component handles local state, but takes initial
                        isTaskRunning={isTaskRunning}
                        openFileBrowser={openFileBrowser}
                        handleLandingBrowse={handleLandingBrowse}
                        showNotification={showNotification}
                        taskProgress={taskProgress}
                        setIsTaskRunning={setIsTaskRunning}
                    />
                )}
                {activeTool === 'sampling' && (
                    <DatasetSampling
                        initialPath={datasetPath}
                        projectConfig={projectConfig}
                        onBrowse={handleLandingBrowse}
                        showNotification={showNotification}
                        isTaskRunning={isTaskRunning}
                        taskProgress={taskProgress}
                        onStartTask={() => setIsTaskRunning(true)}
                    />
                )}
            </main>
        </div>
    );
};

export default DatasetToolsPage;
